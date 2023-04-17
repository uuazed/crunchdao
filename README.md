# Python API for the Crunchdao machine learning tournament
<h1 align="center">

[![PyPi version](https://badgen.net/pypi/v/crunchdao/)](https://pypi.com/project/crunchdao)
</h1>

Interact with the Crunchdao tournament API using Python.

If you encounter a problem or have suggestions, feel free to open an issue.

# Installation

`pip install --upgrade crunchdao`

# Usage

Some actions (like uploading predictions) require an `apikey` to verify
that it is really you interacting with Crunchdao. Keys can be passed to the
Python module as a parameter or you can be set via the `CRUNCHDAO_API_KEY`
environment variable

# Example usage

    import crunchdao
    # some API calls do not require logging in
    client = crunchdao.Client(apikey="foo")
    # download current dataset
    client.download_data(directory=".")
    # get information about your submissions
    submissions = client.submissions()
    print(submissions)  # this is a pandas Dataframe
    # get configure of the current dataset
    client.dataset_config()
    # upload predictions
    predictions = ....  # pandas DataFrame containing your predictions
    submission_id = client.upload(predictions)
    # set comment for the submission, to remember which model that is etc
    client.set_comment(submission_id, "Great model, learning_rate=0.01")
    # Get your resolved targets live score on master dataset
    client.get_scores(dataset=11, resolved_scores=True)