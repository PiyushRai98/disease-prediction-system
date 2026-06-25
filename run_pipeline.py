"""
Main Pipeline Runner for Disease Prediction System.
Executes the complete ML pipeline: data loading, preprocessing,
training, evaluation, and explainability for all datasets.

Usage:
    python run_pipeline.py
    python run_pipeline.py --dataset heart
    python run_pipeline.py --no-tune
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_dataset
from src.preprocessing import PreprocessingPipeline
from src.train import train_all_models, get_training_summary
from src.evaluate import (
    evaluate_all_models,
    create_comparison_table,
    generate_all_plots,
    get_best_model,
)
from src.explainability import generate_explainability_report
from src.utils import logger, save_results
from src.config import MODEL_RESULTS_DIR


def run_pipeline(
    dataset_name: str,
    tune_hyperparameters: bool = True,
    run_explainability: bool = True,
) -> dict:
    """
    Run the complete ML pipeline for a single dataset.

    Args:
        dataset_name: Name of the dataset ('heart', 'diabetes', 'cancer').
        tune_hyperparameters: Whether to perform GridSearchCV.
        run_explainability: Whether to generate SHAP analysis.

    Returns:
        Dictionary with all pipeline results.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING PIPELINE: {dataset_name.upper()}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Step 1: Load data
    logger.info("Step 1: Loading dataset...")
    df = load_dataset(dataset_name)
    logger.info(f"Dataset shape: {df.shape}")

    # Step 2: Preprocess
    logger.info("Step 2: Preprocessing...")
    pipeline = PreprocessingPipeline()
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)
    feature_names = pipeline.feature_names
    logger.info(f"Features: {len(feature_names)}")

    # Step 3: Train models
    logger.info("Step 3: Training models...")
    models = train_all_models(
        X_train,
        y_train,
        dataset_name,
        tune_hyperparameters=tune_hyperparameters,
    )

    # Print training summary
    summary = get_training_summary(models)
    logger.info(f"\nTraining Summary:\n{summary.to_string()}")

    # Step 4: Evaluate models
    logger.info("Step 4: Evaluating models...")
    results = evaluate_all_models(models, X_test, y_test, dataset_name)

    # Print comparison
    comparison = create_comparison_table(results)
    logger.info(f"\nModel Comparison:\n{comparison.to_string()}")

    # Best model
    best = get_best_model(results)
    logger.info(f"\n🏆 Best model: {best}")

    # Step 5: Generate plots
    logger.info("Step 5: Generating plots...")
    generate_all_plots(results, y_test, dataset_name)

    # Step 6: Explainability
    if run_explainability:
        logger.info("Step 6: Generating explainability report...")
        try:
            explainability_report = generate_explainability_report(
                models, X_train, X_test, feature_names, dataset_name
            )
        except Exception as e:
            logger.warning(f"Explainability error (non-fatal): {e}")
            explainability_report = {}

    elapsed = time.time() - start_time
    logger.info(f"\nPipeline complete for {dataset_name} in {elapsed:.1f}s")

    return {
        "dataset": dataset_name,
        "best_model": best,
        "comparison": comparison.to_dict(),
        "elapsed_time": elapsed,
    }


def main():
    """Main entry point for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Disease Prediction System - Training Pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["heart", "diabetes", "cancer", "all"],
        default="all",
        help="Dataset to process (default: all)",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning for faster execution",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Skip SHAP explainability analysis",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DISEASE PREDICTION SYSTEM - TRAINING PIPELINE")
    logger.info("=" * 60)

    datasets = (
        ["heart", "diabetes", "cancer"]
        if args.dataset == "all"
        else [args.dataset]
    )

    all_results = {}
    total_start = time.time()

    for dataset_name in datasets:
        try:
            result = run_pipeline(
                dataset_name=dataset_name,
                tune_hyperparameters=not args.no_tune,
                run_explainability=not args.no_explain,
            )
            all_results[dataset_name] = result
        except Exception as e:
            logger.error(f"Pipeline failed for {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}

    # Final summary
    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE - FINAL SUMMARY")
    logger.info("=" * 60)

    for name, result in all_results.items():
        if "error" in result:
            logger.error(f"  {name}: FAILED - {result['error']}")
        else:
            logger.info(
                f"  {name}: Best Model = {result['best_model']} "
                f"({result['elapsed_time']:.1f}s)"
            )

    logger.info(f"\nTotal execution time: {total_time:.1f}s")
    logger.info("Models saved to: models/")
    logger.info("Results saved to: reports/model_results/")
    logger.info("Figures saved to: reports/figures/")
    logger.info("\nTo launch the web app: streamlit run app/streamlit_app.py")

    # Save overall results
    save_results(
        {k: v for k, v in all_results.items() if "error" not in v},
        MODEL_RESULTS_DIR / "pipeline_results.json",
    )


if __name__ == "__main__":
    main()
