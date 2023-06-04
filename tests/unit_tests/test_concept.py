import logging
import pytest
import os
import pickle
from modules.models.template import build, load_training_data, load_test_data

logging.basicConfig(
    filename="test_log.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


@pytest.fixture
def model():
    model = build(None)
    assert model is not None, "Model is not initialized"
    return model


@pytest.fixture
def data():
    training_data = load_training_data("csv/")
    test_data = load_test_data("csv/")
    return {"train": training_data, "val": test_data}


# Algorithm Verification Tests
def test_model_train(model, data):
    try:
        train_data, train_label = data["train"]["data"], data["train"]["label"]
        history = model.fit(train_data, train_label, epochs=1, batch_size=32)
        assert history is not None, "Model training failed"
    except Exception as e:
        logging.error(f"Model training failed with error: {str(e)}")
        raise


# Model Performance Tests
def test_model_performance(model, data):
    try:
        test_data, test_label = data["val"]["data"], data["val"]["label"]
        performance = model.evaluate(test_data, test_label)
        logging.info(f"Model performance: {performance}")

        assert performance[1] >= 0.8, "Model performance is below 0.8"

        best_performance = 0.0
        last_version = 0
        if os.path.exists("best_performance.pkl"):
            with open("best_performance.pkl", "rb") as f:
                best_performance, last_version = pickle.load(f)

        if performance[1] > best_performance:
            with open("best_performance.pkl", "wb") as f:
                pickle.dump((performance[1], last_version + 1), f)
            model.save(f"../workbench/models/AVMotorcycles_v{last_version + 1}.h5")

    except Exception as e:
        logging.error(f"Model performance test failed with error: {str(e)}")
        raise
