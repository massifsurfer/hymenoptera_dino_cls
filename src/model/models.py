import pytorch_lightning as L
import timm
import torch
import torch.nn as nn
from torchmetrics import AUROC, Accuracy, ConfusionMatrix, F1Score, Precision, Recall


class HymenopteraClassifier(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.threshold = cfg.model.threshold

        self.backbone = timm.create_model(
            cfg.model.architecture.backbone,
            pretrained=cfg.model.architecture.pretrained,
            features_only=cfg.model.architecture.features_only,
        )

        if cfg.model.architecture.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.randn(
                1, 3, cfg.dataset.transforms.img_size, cfg.dataset.transforms.img_size
            )
            dummy_tokens = self.backbone.forward_features(dummy_input)

            self.feature_dim = dummy_tokens.shape[-1]
            self.num_tokens = dummy_tokens.shape[1]

        self.classifier = nn.Linear(3 * self.feature_dim, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

        self.criterion = nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy(task=cfg.model.task)
        self.val_acc = Accuracy(task=cfg.model.task)
        self.val_precision = Precision(task=cfg.model.task)
        self.val_recall = Recall(task=cfg.model.task)
        self.val_f1 = F1Score(task=cfg.model.task)
        self.val_auroc = AUROC(task=cfg.model.task)
        self.val_confmatrix = ConfusionMatrix(task=cfg.model.task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)

        cls_token = tokens[:, 0, :]  # [B, feature_dim]
        patch_tokens = tokens[
            :, self.backbone.num_prefix_tokens :, :
        ]  # [B, num_patches, feature_dim]

        avg_patch = patch_tokens.mean(dim=1)  # [B, feature_dim]
        max_patch = patch_tokens.max(dim=1)[0]  # [B, feature_dim]

        # CLS + Avg + Max concat
        combined = torch.cat(
            [cls_token, avg_patch, max_patch], dim=1
        )  # [B, 3 * feature_dim]

        return self.classifier(combined).flatten()  # [B,]

    def _compute_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor, stage: str = "val"
    ):
        labels = labels.float()
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).long()

        metrics = {
            f"{stage}_loss": self.criterion(logits, labels),
            f"{stage}_acc": self.val_acc(preds, labels),
            f"{stage}_precision": self.val_precision(preds, labels),
            f"{stage}_recall": self.val_recall(preds, labels),
            f"{stage}_f1": self.val_f1(preds, labels),
            f"{stage}_auroc": self.val_auroc(probs, labels),
        }
        return metrics

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels.float())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            self.train_acc((torch.sigmoid(logits) > self.threshold).long(), labels),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        metrics = self._compute_metrics(logits, labels, "val")

        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_epoch=True,
                prog_bar=(name in ["val_loss", "val_acc", "val_f1"]),
            )

        self.val_confmatrix((torch.sigmoid(logits) > self.threshold).long(), labels)
        return metrics["val_loss"]

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels.float())

        preds = (torch.sigmoid(logits) > self.threshold).long()
        acc = (preds == labels).float().mean()

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.model.optimizer.lr,
            weight_decay=self.cfg.model.optimizer.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.model.max_epochs, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
