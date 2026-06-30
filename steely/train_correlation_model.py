from .correlation_signal_classifier import CorrelationSignalClassifier
from .word_correlations import (
    CorrelationMethod,
    stem_tokenise,
    texts_to_word_correlations,
)
import polars as pl

if __name__ == "__main__":
    from argparse import ArgumentParser

    ROOT_DIR = "."
    DATA_DIR = ROOT_DIR + "/" + "data"
    DATA_TASK_1_DIR = DATA_DIR  + "/" +  "pan25-generative-ai-detection-task1-train"    
    parser = ArgumentParser(
        description="Run inference on the CorrelationSignalClassifier model."
    )
    parser.add_argument(
        "output_model", type=str, help="Directory to save the output predictions."
    )
    parser.add_argument(
        "--n_gram", type=int, default=1, help="N-gram size for the classifier."
    )
    parser.add_argument(
        "--correlation-method",
        type=str,
        default="pearson",
        choices=[m.value for m in CorrelationMethod],
        help="Correlation method to use (pearson, spearman, jaccard)",
    )

    args = parser.parse_args()

    train_df = pl.read_ndjson(DATA_TASK_1_DIR + "/" +  "train.jsonl")
    print(f"Using word correlations from {ROOT_DIR + "/" +  'tmp'}")
    word_correlations = texts_to_word_correlations(
        train_df,
        CorrelationMethod(args.correlation_method),
        n_gram=3,
        word_correlations_path=ROOT_DIR + "/" +  "tmp",
        vectorized_texts_path=ROOT_DIR + "/" +  "tmp",
    )
    clf = CorrelationSignalClassifier(word_correlations, n_gram=args.n_gram)
    clf.train(train_df["text"].to_list(), train_df["label"].to_numpy())
    clf.save_to_json(args.output_model)