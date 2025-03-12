import pandas as pd
import re
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path


class Analyser:
    """
    A comprehensive analyzer for text classification tasks.

    This class handles data loading, preprocessing, and model evaluation
    for binary text classification problems, with special handling for
    the LIAR dataset format.
    """

    def __init__(self, random_state: int = 42, log_level: int = logging.INFO):
        """
        Initialize the analyzer with configuration parameters.

        Args:
            random_state: Seed for reproducibility
            log_level: Logging level to use
        """
        self.train = None
        self.test = None
        self.random_state = random_state

        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.

        Args:
            text: Raw text input

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            self.logger.warning(f"Non-string input to clean_text: {type(text)}")
            text = str(text)

        text = text.lower()
        text = re.sub(r'\W', ' ', text)  # Remove non-word characters
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
        text = text.strip()
        return text

    def prep_liar_data(self,
                       train_path: Union[str, Path],
                       validation_path: Union[str, Path],
                       test_path: Union[str, Path],
                       columns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess data from the LIAR dataset format.

        Args:
            train_path: Path to training data file
            validation_path: Path to validation data file
            test_path: Path to test data file
            columns: Column names for the dataset (if None, will use default LIAR columns)

        Returns:
            Tuple of (train_data, test_data)
        """
        # Default columns for LIAR dataset if not provided
        if columns is None:
            columns = [
                'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
                'state_info', 'party_affiliation', 'barely_true_counts',
                'false_counts', 'half_true_counts', 'mostly_true_counts',
                'pants_on_fire_counts', 'context'
            ]

        try:
            # Load datasets
            test_df = pd.read_csv(test_path, sep='\t', header=None, names=columns)
            train_df = pd.read_csv(train_path, sep='\t', header=None, names=columns)
            valid_df = pd.read_csv(validation_path, sep='\t', header=None, names=columns)

            # Combine train and validation sets
            train_df = pd.concat([train_df, valid_df], ignore_index=True)

            # Select relevant columns
            self.train = train_df[['statement', 'label']].copy()
            self.test = test_df[['statement', 'label']].copy()

            # Create binary labels: 1 for true-ish, 0 for false-ish
            label_mapping = {
                "true": 1,
                "mostly-true": 1,
                "half-true": 1,
                "barely-true": 0,
                "false": 0,
                "pants-fire": 0
            }

            # Apply label mapping
            self.train.loc[:, 'label'] = self.train['label'].map(label_mapping)
            self.test.loc[:, 'label'] = self.test['label'].map(label_mapping)

            # Apply text cleaning
            self.train.loc[:, 'statement'] = self.train['statement'].apply(self.clean_text)
            self.test.loc[:, 'statement'] = self.test['statement'].apply(self.clean_text)

            # Convert to proper types
            self.train['label'] = self.train['label'].astype(int)
            self.test['label'] = self.test['label'].astype(int)

            # Log data statistics
            self.logger.info(f"Training data shape: {self.train.shape}")
            self.logger.info(f"Test data shape: {self.test.shape}")
            self.logger.info(f"Label distribution in training: {self.train['label'].value_counts(normalize=True)}")

            return self.train, self.test

        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def evaluate_model(self, y_pred, y_test, X_test, model, class_names=None,
                       output_path: Optional[Union[str, Path]] = None,
                       save_plots: bool = False):
        """
        Comprehensive model evaluation with metrics and visualizations.

        Args:
            model: Trained model with predict and predict_proba methods
            X_test: Test features
            y_test: True test labels
            class_names: Names for the classes (optional)
            output_path: Where to save plots (optional)
            save_plots: Whether to save plots to disk

        Returns:
            Dict of evaluation metrics
        """

        # Get predicted probabilities for ROC curve

        y_pred_proba = model.predict_proba(X_test)[:, 1]


        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Output basic metrics
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        class_labels = class_names if class_names else ["Class 0", "Class 1"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=axes[0], cmap='Blues', values_format='.2f')
        axes[0].set_title('Normalized Confusion Matrix')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(alpha=0.3)

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        axes[2].plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall Curve')
        axes[2].legend(loc="lower left")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()

        if save_plots and output_path:
            path = Path(output_path)
            path.mkdir(exist_ok=True, parents=True)
            plt.savefig(path / 'model_evaluation.png', dpi=300, bbox_inches='tight')

        plt.show()

        # Return all metrics as a dictionary
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'average_precision': avg_precision
        }

        return metrics

    def get_data_split(self):
        """
        Return the current train/test data split.

        Returns:
            Tuple containing (X_train, y_train, X_test, y_test)
        """
        if self.train is None or self.test is None:
            raise ValueError("Data has not been prepared. Call prep_liar_data first.")

        X_train = self.train['statement']
        y_train = self.train['label']
        X_test = self.test['statement']
        y_test = self.test['label']

        return X_train, y_train, X_test, y_test



