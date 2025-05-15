#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to calculate and visualize hyperparameter importances from an Optuna study
using fANOVA and save results to CSV and plot files.

Dependencies:
    - pandas            # For data manipulation and CSV handling
    - matplotlib        # For plotting
    - seaborn           # For enhanced plotting
    - optuna            # Main package for hyperparameter optimization
    - psycopg2-binary   # PostgreSQL adapter (required for database connection)
    - sqlalchemy        # Required for database connection
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.importance import FanovaImportanceEvaluator
import argparse
import logging
from typing import Dict


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate hyperparameter importance using fANOVA from an Optuna study'
    )
    parser.add_argument('--study-name', '-s', dest='study_name', type=str, default="",
                        help='Name of the Optuna study')
    
    # Database connection arguments
    db_group = parser.add_argument_group('Database connection options')
    db_group.add_argument('--url', dest='db_url', type=str,
                        help='Complete database URL (if provided, other DB options are ignored)')
    db_group.add_argument('--host', '-a', dest='db_host', type=str,
                        default="pg-windforecasting-aiven-wind-forecasting.e.aivencloud.com",
                        help='Database host')
    db_group.add_argument('--port', dest='db_port', type=int, default=12472,
                        help='Database port')
    db_group.add_argument('--name', '-db', dest='db_name', type=str, default="defaultdb",
                        help='Database name')
    db_group.add_argument('--user', '-u', dest='db_user', type=str, default="avnadmin",
                        help='Database user')
    db_group.add_argument('--password', '-p', dest='db_password', type=str,
                        help='Database password (required if --url is not provided)')
    
    evaluator_group = parser.add_argument_group('Evaluator options')
    evaluator_group.add_argument('--n-trees', dest='n_trees', type=int, default=64,
                        help='Number of trees for fANOVA (default: 64)')
    evaluator_group.add_argument('--max-depth', dest='max_depth', type=int, default=64,
                        help='Maximum depth of trees for fANOVA (default: 64)')
    evaluator_group.add_argument('--seed', dest='seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Output options
    parser.add_argument('--output-dir', '-o', dest='output_dir', type=str,
                        default="./results",
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.db_url and not args.db_password:
        parser.error("Either --url or --db-password must be provided")
    
    return args


def get_database_url(args):
    """Get the database URL from the arguments."""
    if args.db_url:
        # Ensure protocol is postgresql:// not postgres://
        url = args.db_url
        if url.startswith("postgres://"):
            url = "postgresql" + url[8:]
        return url
    
    # Create the DB URL based on provided components
    return (
        f"postgresql://{args.db_user}:{args.db_password}@{args.db_host}:"
        f"{args.db_port}/{args.db_name}?sslmode=require"
    )


def load_optuna_study(args, logger):
    """Load an Optuna study from the database."""
    logger.info(f"Loading study '{args.study_name}' from database...")
    
    # Get the database URL
    db_url = get_database_url(args)
    logger.info(f"Using connection URL format: {db_url.split(':', 1)[0]}://*****")
    
    # Load the study
    try:
        study = optuna.load_study(study_name=args.study_name, storage=db_url)
        logger.info(f"Successfully loaded study with {len(study.trials)} trials")
        return study
    except Exception as e:
        logger.error(f"Failed to load study: {e}")
        if "Can't load plugin: sqlalchemy.dialects:postgres" in str(e):
            logger.error("ERROR: The database URL must start with 'postgresql://' not 'postgres://'")
        raise


def calculate_importances(study, logger, n_trees=64, max_depth=64, seed=42):
    """Calculate hyperparameter importances using fANOVA."""
    logger.info("Calculating hyperparameter importances using fANOVA...")
    
    try:
        # Use Optuna's FanovaImportanceEvaluator to calculate importances
        evaluator = FanovaImportanceEvaluator(
            n_trees=64,
            max_depth=64,
            seed=42
        )
        importances = optuna.importance.get_param_importances(
            study=study,
            evaluator=evaluator
        )
        
        logger.info(f"Successfully calculated importances for {len(importances)} parameters")
        return importances
    except Exception as e:
        logger.error(f"Failed to calculate importances: {e}")
        raise


def plot_importances(importances: Dict[str, float], output_file: str, logger):
    """Create and save a plot of hyperparameter importances using seaborn."""
    logger.info(f"Creating importance plot and saving to {output_file}...")
    
    try:
        # Convert importances to DataFrame for plotting
        df = pd.DataFrame({
            'Parameter': list(importances.keys()),
            'Importance': list(importances.values())
        })
        
        # Sort by importance value (descending)
        df = df.sort_values('Importance', ascending=False)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Fix the deprecation warning by using y parameter as both y and hue
        # but disable the legend to keep the original look
        ax = sns.barplot(
            x='Importance',
            y='Parameter',
            hue='Parameter',  # Add this to fix the warning
            data=df,
            palette='Blues_d',
            legend=False      # Don't show the legend
        )
        
        # Add value labels to the bars
        for i, v in enumerate(df['Importance']):
            ax.text(v + 0.01, i, f"{v:.4f}", va='center')
        
        # Set labels and title
        plt.title('Hyperparameter Importance (fANOVA)', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Parameter', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to create or save plot: {e}")
        raise


def save_importances_to_csv(importances: Dict[str, float], output_file: str, logger):
    """Save hyperparameter importances to a CSV file."""
    logger.info(f"Saving importance values to {output_file}...")
    
    try:
        # Convert importances to DataFrame
        df = pd.DataFrame({
            'Parameter': list(importances.keys()),
            'Importance': list(importances.values())
        })
        
        # Sort by importance value (descending)
        df = df.sort_values('Importance', ascending=False)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"CSV saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        raise


def main():
    """Main execution function."""
    # Set up logging
    logger = setup_logging()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output file paths
    plot_file = os.path.join(args.output_dir, f"{args.study_name}_param_importances.png")
    csv_file = os.path.join(args.output_dir, f"{args.study_name}_param_importances.csv")
    
    # Load the study
    study = load_optuna_study(args, logger)
    
    # Calculate importances
    importances = calculate_importances(study, logger, n_trees=args.n_trees, max_depth=args.max_depth, seed=args.seed)

    # Create and save the plot
    plot_importances(importances, plot_file, logger)
    
    # Save importances to CSV
    save_importances_to_csv(importances, csv_file, logger)
    
    logger.info("Script completed successfully!")


if __name__ == "__main__":
    main()