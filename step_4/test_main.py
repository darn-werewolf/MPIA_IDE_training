import os
import pytest
from PIL import Image

# Adjust the import according to your script's structure

# Define test input and output directories
TEST_INPUT_DIR = "./test_data/test_input/"
TEST_OUTPUT_DIR = "./test_data/test_output/"


# Setup and teardown functions to prepare test environment
@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    # Set environment variables for the test
    os.environ["DATA_INPUT_LOCATION"] = TEST_INPUT_DIR
    os.environ["DATA_OUTPUT_LOCATION"] = TEST_OUTPUT_DIR

    # Setup: Create necessary directories and a test file if not exist
    os.makedirs(TEST_INPUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    test_image_path = os.path.join(TEST_INPUT_DIR, "test_image.png")
    if not os.path.exists(test_image_path):
        # Create a simple image for testing
        test_image = Image.new("RGB", (100, 100), color="red")
        test_image.save(test_image_path)

    # Yield to test execution
    yield

    # Teardown: Optionally remove test output files
    output_files = os.listdir(TEST_OUTPUT_DIR)
    for f in output_files:
        os.remove(os.path.join(TEST_OUTPUT_DIR, f))


def test_process_image():
    from main import (
        process_image,
    )

    process_image("test_image")  # Assuming 'test_image' is the name without extension

    # Verify that output files are created in the specified output location
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "test_image.png"))
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "test_image.txt"))
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "test_image_deadpix.pkl"))

    # Additional checks could include verifying the contents of these files,
    # but this simple existence check is a good starting point for an end-to-end test.
