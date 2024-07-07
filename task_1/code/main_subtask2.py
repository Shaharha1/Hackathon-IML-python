from argparse import ArgumentParser
import logging
from hackathon_code.preprocess2 import *
from hackathon_code.model1 import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    # 2. preprocess the training set
    logging.info("preprocessing train...")
    preprocess_train_path = preprocess_task_2(args.training_set, False)
    target_feature = "total_time"

    # 3. train a model
    model = LinearRegModel(0)
    logging.info("training...")
    model.fit(preprocess_train_path, target_feature)


    # 4. load the test set (args.test_set)
    # 5. preprocess the test set
    logging.info("preprocessing test...")
    preprocess_test_path, unique_id = preprocess_task_2(args.test_set, True)
    test_set = pd.read_csv(preprocess_test_path, encoding="ISO-8859-8")
    x_label = pd.read_csv(unique_id, encoding="ISO-8859-8")


    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = model.predict(test_set)

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    output = pd.DataFrame({
        'trip_id_unique': x_label['trip_id_unique'],
        'trip_duration_in_minutes': predictions.astype(int)})
    output.to_csv(args.out, index=False)