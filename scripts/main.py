import argparse
import logging
import mlflow

logger = logging.getLogger(__name__)


def main(raw_data, processed_data, model_path, model_name, run_id):

    run_id = mlflow.active_run().info.run_id if run_id is None else run_id

    mlflow.run(
        '.',
        entry_point='data_loading',
        parameters={'output_path': raw_data},
        use_conda=False,
        run_id=run_id
    )
    mlflow.run(
        '.',
        entry_point='pre_processing',
        parameters={'input_path': raw_data, 'output_path': processed_data},
        use_conda=False,
        run_id=run_id
    )
    mlflow.run(
        '.',
        entry_point='train_model',
        parameters={'input_path': processed_data, 'model_path': model_path, 'model_name': model_name},
        use_conda=False,
        run_id=run_id
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data', help='path where raw datasets are saved')
    parser.add_argument('--processed-data', help='path where processed datasets are saved')
    parser.add_argument('--model-path', help='path where trained model is saved')
    parser.add_argument('--model-name', help='model name')
    parser.add_argument('--run-id', help='MLflow run id', default=None, required=False)
    args = parser.parse_args()

    mlflow.start_run()

    main(**vars(args))
