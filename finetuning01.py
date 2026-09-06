from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
import torch
from sklearn.metrics.pairwise import paired_cosine_distances


class FineTuneEmbeddingModel:
    def __init__(self):
        self.model = self.setup_model()
        self.dataset = self.load_dataset()
        self.train_dataset = self.create_train_dataset()
        self.test_dataset = self.create_test_dataset()
        self.eval_dataset = self.create_eval_dataset()
    
        self.dev_evaluator = self.setup_dev_evaluator()
        self.loss = self.setup_loss()
        self.args = self.setup_training_args()
        self.trainer = self.setup_trainer()

    def setup_model(self):
        return SentenceTransformer(
            "microsoft/mpnet-base",
            model_card_data=SentenceTransformerModelCardData(
                language="en",
                license="apache-2.0",
                model_name="MPNet base trained on AllNLI triplets",
            )
        )

    def load_dataset(self):
        return load_dataset("sentence-transformers/all-nli", "triplet")
    
    def create_train_dataset(self):
        return self.dataset["train"].select(range(1000))

    def create_test_dataset(self):
        return self.dataset["test"].select(range(200))

    def create_eval_dataset(self):
        return self.dataset["dev"].select(range(200))

    def setup_dev_evaluator(self):
        return TripletEvaluator(
            anchors=self.eval_dataset["anchor"],
            positives=self.eval_dataset["positive"],
            negatives=self.eval_dataset["negative"],
            name="all-nli-dev",
        )

    def setup_loss(self):
        return MultipleNegativesRankingLoss(self.model)

    def setup_training_args(self):
        return SentenceTransformerTrainingArguments(
            output_dir="models/mpnet-base-all-nli-triplet",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_ratio=0.1,
            fp16=False,  # Set to False if GPU can't handle FP16
            bf16=False,  # Set to True if GPU supports BF16
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            logging_steps=100,
            run_name="mpnet-base-all-nli-triplet",  # Used in W&B if `wandb` is installed
        )

    def setup_trainer(self):
        return SentenceTransformerTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            loss=self.loss,
            evaluator=self.dev_evaluator,
        )

    def evaluate_cosine_accuracy(self, dataset):
        self.model.eval()
        anchors = dataset["anchor"]
        positives = dataset["positive"]
        negatives = dataset["negative"]

        with torch.no_grad():
            anchor_embeddings = self.model.encode(anchors, convert_to_numpy=True)
            positive_embeddings = self.model.encode(positives, convert_to_numpy=True)
            negative_embeddings = self.model.encode(negatives, convert_to_numpy=True)

        pos_cos_distances = paired_cosine_distances(anchor_embeddings, positive_embeddings)
        neg_cos_distances = paired_cosine_distances(anchor_embeddings, negative_embeddings)

        correct = 0
        for pos_dist, neg_dist in zip(pos_cos_distances, neg_cos_distances):
            if pos_dist < neg_dist:
                correct += 1
        accuracy = correct / len(pos_cos_distances)
        return accuracy

    def train_model(self):
        self.trainer.train()

if __name__ == "__main__":
    fine_tune_embedding = FineTuneEmbeddingModel()

    eval_before = fine_tune_embedding.evaluate_cosine_accuracy(fine_tune_embedding.test_dataset)
    print(f"Triplet accuracy on the held-out test split BEFORE fine-tuning: {eval_before:.4f}")

    fine_tune_embedding.train_model()

    eval_after = fine_tune_embedding.evaluate_cosine_accuracy(fine_tune_embedding.test_dataset)
    print(f"Triplet accuracy on the held-out test split AFTER  fine-tuning: {eval_after:.4f}")
    print(f"Absolute improvement: {eval_after - eval_before:+.4f}")
