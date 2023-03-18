import os
import logging
from typing import Dict, List, Optional

import requests
import pandas as pd
import inflection

from crunchdao import utils


BASE_URL = "https://api.tournament.crunchdao.com"


logger = logging.getLogger(__name__)


class Client:
    """"Python API for the Crunchdao machine learning tournament"""

    def __init__(self, apikey: str = None):
        """initialize the client

        Args:
            apikey (str, optional): your crunchdao API key
        """
        if apikey is None:
            # try getting apikey from env variable
            self.apikey = os.getenv("CRUNCHDAO_API_KEY")
        else:
            self.apikey = apikey

    def raw_request(self, url: str, params: Dict = None,
                    authorization: bool = False) -> Dict:
        """Send a raw request to the crunchdao API.

        Args:
            url (str)
            params (dict, optional): dict of variables
            authorization (bool, optional): does the request require
                authorization, defaults to `False`

        Returns:
            dict: Result of the request

        Raises:
            ValueError: if something went wrong with the requests. For example,
                this could be a missing apikey or a problem at
                crunchdao's end. Have a look at the error messages, in most
                cases the problem is obvious.
        """
        if params is None:
            params = {}
        if authorization:
            if self.apikey is None:
                raise ValueError("api key needed for this request")
            params["apiKey"] = self.apikey
        response = requests.get(url, params=params)
        # FIXME add error handling
        return response.json()

    def download_data(self, directory: str = ".") -> List[str]:
        """Download training data, targets and test data

        Args:
            directory (str): directory where the files are downloaded to

        Returns:
            list[str]: Paths to the three files

        Example:
            >>> client = crunchdao.Client()
            >>> client.download_data()
            ['./X_train.csv', './y_train.csv', './X_test.csv']
        """
        paths = []
        for filename in ["X_train.csv", "y_train.csv", "X_test.csv"]:
            url = f'https://tournament.crunchdao.com/data/{filename}'
            path = utils.download_file(url, os.path.join(directory, filename))
            paths.append(path)
        return paths

    def set_comment(self, submission_id: int, comment: str) -> None:
        """Set comment of a submission

        Args:
            submission_id (int): ID of the relevant submission
            comment (str)

        Example:
            >>> client = crunchdao.Client()
            >>> submission_id = client.upload(...)
            >>> client.set_comment(submission_id, "bla bla")
            Comment set.
        """
        response = requests.patch(
            f"{BASE_URL}/v2/submissions/{submission_id}",
            json={"comment": comment},
            params={"apiKey": self.apikey})
        if response.status_code == 200:
            logger.info("Comment set.")
        else:
            logger.error("setting comment failed")
            logger.error(response.content)

    def upload(self, predictions: pd.DataFrame) -> Optional[int]:
        """Upload predictions to the tournament

        Args:
            predictions (pd.DataFrame): dataframe with your predictions

        Returns:
            int: ID of the submission

        Example:
            >>> client = crunchdao.Client()
            >>> predictions = .... # pd.DataFrame containing your predictions
            >>> client.upload(predictions)
        """
        response = requests.post(
            BASE_URL + "/v2/submissions",
            files={"file": ("x", predictions.to_csv(index=False).encode('ascii'))},
            data={"apiKey": self.apikey})

        if response.status_code == 200:
            logger.info("Submission submitted :)")
            submission_id = response.json()["id"]
            return submission_id
        elif response.status_code == 423:
            logger.error("Submissions are close")
            logger.info("You can only submit during rounds eg: "
                        "Friday 7pm GMT+1 to Sunday midnight GMT+1.")
            logger.info("Or the server is currently crunching the submitted "
                        "files, please wait some time before retrying.")
        elif response.status_code == 422:
            logger.error("API Key is missing or empty")
            logger.info("Did you forget to fill the API_KEY variable?")
        elif response.status_code == 404:
            logger.error("Unknown API Key")
            logger.info("You should check that the provided API key is valid "
                        "and is the same as the one you've received by email.")
        elif response.status_code == 400:
            logger.error("The file must not be empty")
            logger.info("You have send a empty file.")
        elif response.status_code == 401:
            logger.error("Your email hasn't been verified")
            logger.info("Please verify your email or contact a cruncher.")
        elif response.status_code == 403:
            logger.error("Access Denied")
            logger.info("Please setup your API_KEY.")
        elif response.status_code == 409:
            logger.error("Duplicate submission")
            logger.info("Your work has already been submitted with the exact "
                        "same results, if you think this a false positive, "
                        "contact a cruncher.")
            logger.info("MD5 collision probability: 1/2^128 (source: "
                        "https://stackoverflow.com/a/288519/7292958)")
        elif response.status_code == 429:
            logger.error("Too many submissions")
        else:
            logger.error(f"Server returned: {response.status_code}")
            logger.info("Ouch! It seems that we were not expecting this kind "
                        "of result from the server, if the probleme persist, "
                        "contact a cruncher.")

    def submissions(self, user_id: int = None,
                    round_num: int = None) -> pd.DataFrame:
        """Get all the details about individual submissions.

        Args:
            user_id (int, optional): selected user_id, defaults to your own
            round_num (int): allows to spefify a single round, defaults to all
                             rounds

        Returns:
            pd.DataFrame: submissions information with the following columns:
                * user_id (`int`)
                * username (`str`)
                * deleted (`bool`)
                * administrator (`bool`)
                * status (`str`)
                * upload_ts (`datetime`)
                * eval_ts (`datetime`)
                * comment (`str`)
                * message (`str`)
                * trace (`str`)
                * file_hash (`str`)
                * file_name (`str`)
                * file_length (`int`)

        Example:
            >>> crunchdao.Client().submissions(round_num=89)
        """
        if user_id is None:
            user_id = "@me"
            authorization = True
        else:
            authorization = False

        url = f"{BASE_URL}/v2/users/{user_id}/submissions"
        params = {}
        if round_num:
            params["round"] = round_num
        data = self.raw_request(url, params, authorization=authorization)

        user = pd.DataFrame([item["user"] for item in data])
        user.rename(columns={"id": "user_id"}, inplace=True)

        file_meta = pd.DataFrame([item["fileMetadata"] for item in data])
        columns = {col: "file_" + col for col in file_meta}
        file_meta.rename(columns=columns, inplace=True)

        general = pd.DataFrame.from_dict(data)
        general.rename(columns={"uploadedAt": "upload_ts",
                                "evaluatedAt": "eval_ts"}, inplace=True)
        general.drop(["user", "userId", "fileMetadata"],
                     axis=1, inplace=True, errors="ignore")
        df = pd.concat([user, general, file_meta], axis=1)
        df.set_index("id", inplace=True)

        # convert CamelCase to snake_case
        df.columns = [inflection.underscore(col) for col in df.columns]
        return df

    def dataset_config(self, round_num: int = None) -> Dict:
        """Get the dataset configuration for some round

        Args:
            round_num (int): allows to spefify a single round, defaults to the
                             latest round

        Returns:
            dictionary with the following fields:
                * dataset_id (`int`)
                * round_id (`int`)
                * live (`bool`)
                * updated (`bool`)
                * periods (`dict`)
                    * red (`str`)
                    * green (`str`)
                    * blue (`str`)
                * inception (`str`)
                * first_of_inception (`bool`)
                * forced_start (`bool`, optional)
                * moons_duration (`str`)
                * negative_prevented (`bool`)
                * dataset_name (`str`)

        Example:
            >>> crunchdao.Client().dataset_config(76)
            {'dataset_id': 4, 'round_id': 76, 'live': False, 'updated': True,
             'periods': {'red': 'P30D', 'green': 'P60D', 'blue': 'P90D'},
             'inception': None, 'first_of_inception': False,
             'forced_start': None, 'moons_duration': 'P7D',
             'negative_prevented': False, 'dataset_name': 'e-kinetic'}

        """
        if round_num is None:
            round_num = "@latest"
        url = f"{BASE_URL}/v2/rounds/{round_num}/dataset-config"
        data = self.raw_request(url)
        data["dataset_id"] = data["dataset"]["id"]
        data["dataset_name"] = data["dataset"]["name"]

        for item in ["id", "dataset"]:
            del data[item]

        cleaned = {inflection.underscore(key): val
                   for key, val in data.items()}
        return cleaned


if __name__ == "__main__":
    client = Client()
    print(client.dataset_config())
