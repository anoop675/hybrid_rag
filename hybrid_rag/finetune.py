from .colbert import ColBERTReranker

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import random


class FineTuneColBERTReranker(ColBERTReranker):
    """Freeze the BERT backbone and train only the ColBERT linear projection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

    def setup_training(self, learning_rate: float = 1e-4, total_steps: int = 100, warmup_ratio: float = 0.1):
        if self.model is None:
            self.load_model()

        for param in self.model.parameters():
            param.requires_grad = False

        if self.linear is not None:
            for param in self.linear.parameters():
                param.requires_grad = True
            train_params = self.linear.parameters()
            trainable_params = sum(p.numel() for p in self.linear.parameters())
        else:
            for param in self.model.encoder.layer[-1].parameters():
                param.requires_grad = True
            train_params = (p for p in self.model.parameters() if p.requires_grad)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Backbone frozen ({total_params:,} params locked)")
        print(f"Trainable params: {trainable_params:,} (linear projection only)")

        class InfoNCELoss(torch.nn.Module):
            def __init__(self, temperature: float = 0.07):
                super().__init__()
                self.temperature = temperature

            def forward(self, pos_scores, neg_scores):
                logits = torch.stack([pos_scores, neg_scores], dim=1) / self.temperature
                labels = torch.zeros(len(pos_scores), dtype=torch.long, device=pos_scores.device)
                return torch.nn.functional.cross_entropy(logits, labels)

        temperature = 0.07
        self.criterion = InfoNCELoss(temperature=temperature)
        self.optimizer = optim.AdamW(train_params, lr=learning_rate, weight_decay=0.01)

        warmup_steps = max(1, int(total_steps * warmup_ratio))
        decay_steps = max(1, total_steps - warmup_steps)

        warmup_sched = LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay_sched = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=decay_steps,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, decay_sched],
            milestones=[warmup_steps],
        )

        print(f"Loss: InfoNCE (temperature={temperature})")
        print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay=0.01)")
        print(f"Scheduler: linear warmup ({warmup_steps} steps) -> linear decay ({decay_steps} steps)")

    def train_step(self, query_texts, pos_doc_texts, neg_doc_texts):
        self.model.eval()
        if self.linear is not None:
            self.linear.train()

        self.optimizer.zero_grad()

        query_emb_batch, query_mask_batch = self._encode(query_texts)
        pos_embs, pos_masks = self._encode(pos_doc_texts)
        neg_embs, neg_masks = self._encode(neg_doc_texts)

        pos_scores, neg_scores = [], []
        for i in range(len(query_texts)):
            pos_scores.append(
                self._maxsim_score(query_emb_batch[i], query_mask_batch[i], pos_embs[i], pos_masks[i])
            )
            neg_scores.append(
                self._maxsim_score(query_emb_batch[i], query_mask_batch[i], neg_embs[i], neg_masks[i])
            )

        pos_scores = torch.stack(pos_scores)
        neg_scores = torch.stack(neg_scores)

        loss = self.criterion(pos_scores, neg_scores)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.linear.parameters() if p.requires_grad],
            max_norm=1.0,
        )

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()


def create_training_triplets(
    queries,
    docs,
    qrels,
    bm25_retriever=None,
    doc_map=None,
    num_negatives=1,
    bm25_top_k_for_negatives=None,
):
    triplets = []
    if doc_map is None:
        doc_map = {doc["doc_id"]: doc for doc in docs}

    qrels_by_query = {}
    for qrel_entry in qrels:
        if qrel_entry["relevance"] > 0:
            query_id = qrel_entry["query_id"]
            if query_id not in qrels_by_query:
                qrels_by_query[query_id] = []
            qrels_by_query[query_id].append(qrel_entry["doc_id"])

    all_doc_ids = list(doc_map.keys())

    print("Creating training triplets (hard negatives if a BM25 retriever is provided)...")
    for query_entry in queries:
        query_id = query_entry["query_id"]
        query_text = query_entry["text"]

        positive_ids = qrels_by_query.get(query_id, [])
        if not positive_ids:
            continue

        for pos_doc_id in positive_ids:
            if pos_doc_id not in doc_map:
                continue
            pos_text = f"{doc_map[pos_doc_id]['title']}. {doc_map[pos_doc_id]['body']}"

            negative_ids = []

            if bm25_retriever and bm25_top_k_for_negatives:
                bm25_results = bm25_retriever.retrieve(query_text, top_k=bm25_top_k_for_negatives)
                bm25_candidate_ids = {doc_id for doc_id, _ in bm25_results}
                hard_negatives = list(bm25_candidate_ids - set(positive_ids))

                if len(hard_negatives) > 0:
                    num_hard = min(num_negatives, len(hard_negatives))
                    negative_ids.extend(random.sample(hard_negatives, num_hard))

            if len(negative_ids) < num_negatives:
                exclude = set(positive_ids) | set(negative_ids)
                random_neg_candidates = [doc_id for doc_id in all_doc_ids if doc_id not in exclude]

                num_random = num_negatives - len(negative_ids)
                if len(random_neg_candidates) < num_random:
                    if num_random > 0:
                        print(
                            f"Warning: not enough unique negatives for query {query_id}. "
                            f"Found {len(negative_ids)}/{num_negatives}."
                        )
                    negative_ids.extend(random_neg_candidates)
                else:
                    negative_ids.extend(random.sample(random_neg_candidates, num_random))

            negative_ids = list(set(negative_ids))
            if not negative_ids:
                continue

            for neg_doc_id in negative_ids:
                neg_text = f"{doc_map[neg_doc_id]['title']}. {doc_map[neg_doc_id]['body']}"
                triplets.append({
                    "query": query_text,
                    "positive_doc": pos_text,
                    "negative_doc": neg_text,
                })

    print(f"Created {len(triplets)} training triplets.")
    return triplets


def run_fine_tuning_loop(fine_tuner, training_data, epochs=1, batch_size=4, val_ratio=0.15, patience=2):
    from tqdm.auto import tqdm

    unique_queries = list({item["query"] for item in training_data})
    random.shuffle(unique_queries)
    num_val_queries = max(1, int(len(unique_queries) * val_ratio))
    val_query_ids = set(unique_queries[:num_val_queries])
    val_triplets = [item for item in training_data if item["query"] in val_query_ids]
    train_triplets = [item for item in training_data if item["query"] not in val_query_ids]

    steps_per_epoch = max(1, len(train_triplets) // batch_size)
    total_steps = steps_per_epoch * epochs

    print("ColBERT fine-tuning")
    print(f"Queries: {len(unique_queries) - num_val_queries} train / {num_val_queries} val (query-level split)")
    print(f"Triplets: {len(train_triplets)} train / {len(val_triplets)} val")
    print(f"Epochs: {epochs} Batch size: {batch_size}")
    print(f"Opt steps: {total_steps} Early-stop patience: {patience}")

    best_val_loss = float("inf")
    best_checkpoint = None
    epochs_without_improvement = 0
    fine_tuner.best_epoch = 0
    history = []

    for epoch in range(epochs):
        fine_tuner.model.train()
        train_loss_total = 0.0
        num_batches = 0

        progress = tqdm(
            range(0, len(train_triplets), batch_size),
            desc=f"Epoch {epoch + 1}/{epochs} [train]",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for i in progress:
            batch = train_triplets[i:i + batch_size]
            if not batch:
                continue
            loss = fine_tuner.train_step(
                [item["query"] for item in batch],
                [item["positive_doc"] for item in batch],
                [item["negative_doc"] for item in batch],
            )
            train_loss_total += loss
            num_batches += 1
            running_avg = train_loss_total / num_batches
            progress.set_postfix({"loss": f"{running_avg:.4f}"})

        avg_train_loss = train_loss_total / max(1, num_batches)

        fine_tuner.model.eval()
        val_loss_total = 0.0
        num_val_batches = 0

        with torch.no_grad():
            val_progress = tqdm(
                range(0, len(val_triplets), batch_size),
                desc=f"Epoch {epoch + 1}/{epochs} [val]",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for i in val_progress:
                batch = val_triplets[i:i + batch_size]
                if not batch:
                    continue

                query_emb_batch, query_mask_batch = fine_tuner._encode([item["query"] for item in batch])
                pos_embs, pos_masks = fine_tuner._encode([item["positive_doc"] for item in batch])
                neg_embs, neg_masks = fine_tuner._encode([item["negative_doc"] for item in batch])

                pos_scores = torch.stack([
                    fine_tuner._maxsim_score(
                        query_emb_batch[j], query_mask_batch[j], pos_embs[j], pos_masks[j]
                    )
                    for j in range(len(batch))
                ])
                neg_scores = torch.stack([
                    fine_tuner._maxsim_score(
                        query_emb_batch[j], query_mask_batch[j], neg_embs[j], neg_masks[j]
                    )
                    for j in range(len(batch))
                ])

                val_loss = fine_tuner.criterion(pos_scores, neg_scores)
                val_loss_total += val_loss.item()
                num_val_batches += 1
                val_progress.set_postfix({"loss": f"{val_loss_total / num_val_batches:.4f}"})

        avg_val_loss = val_loss_total / max(1, num_val_batches)
        current_lr = fine_tuner.optimizer.param_groups[0]["lr"]

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint = {k: v.clone() for k, v in fine_tuner.model.state_dict().items()}
            fine_tuner.best_epoch = epoch + 1
            epochs_without_improvement = 0
            status = "best"
        else:
            epochs_without_improvement += 1
            status = f"no improvement ({epochs_without_improvement}/{patience})"

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr,
            "status": status,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train={avg_train_loss:.4f} val={avg_val_loss:.4f} "
            f"lr={current_lr:.1e} {status}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} (best was epoch {fine_tuner.best_epoch})")
            break

    if best_checkpoint is not None:
        fine_tuner.model.load_state_dict(best_checkpoint)

    print(f"Fine-tuning complete. Best epoch: {fine_tuner.best_epoch} Best val loss: {best_val_loss:.4f}")
    print("Epoch Train Loss Val Loss LR Status")
    for row in history:
        marker = " (best)" if row["epoch"] == fine_tuner.best_epoch else ""
        print(
            f"{row['epoch']} {row['train_loss']:.4f} {row['val_loss']:.4f} "
            f"{row['lr']:.1e} {row['status']}{marker}"
        )
