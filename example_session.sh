#!/usr/bin/env bash
# usage: ./example_session.sh
set -e

echo "1) create sample datasets"
python dataset_utils.py

echo "2) start server in background (in separate terminal recommended)"
echo "   python server.py"
echo ""
echo "For the demo, open a new terminal and run the following commands:"

echo ""
echo "List datasets:"
echo "python client.py list_datasets"
echo ""
echo "Train a logistic regression on iris:"
echo "python client.py train_model -p dataset=iris.csv -p algorithm=logistic_regression"
echo ""
echo "List models:"
echo "python client.py list_models"
echo ""
echo "Assume model_id from train result was 'logistic_regression_XXXXXXXX', evaluate:"
echo "python client.py evaluate_model -p model_id=logistic_regression_XXXXXXXX"
echo ""
echo "Predict (example input for Iris, 4 features):"
echo "python client.py predict -p model_id=YOUR_MODEL_ID -p input_list='[[5.1,3.5,1.4,0.2]]'"
echo ""
echo "Explain prediction:"
echo "python client.py explain_prediction -p model_id=YOUR_MODEL_ID -p input_row='[5.1,3.5,1.4,0.2]'"
