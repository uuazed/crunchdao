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

    def download_data(self, directory: str = ".",
                      file_format: str = "csv") -> List[str]:
        """Download training data, targets, test data and a submission example

        Args:
            directory (str): directory where the files are downloaded to
            file_format (str): `csv` or 'parquet`

        Returns:
            list[str]: Paths to the three files

        Example:
            >>> client = crunchdao.Client()
            >>> client.download_data()
            ['./X_train.csv', './y_train.csv', './X_test.csv', './example_submission.csv']
        """
        assert file_format in {"csv", "parquet"}, "unknown file format"
        paths = []
        for dataset in ["X_train", "y_train", "X_test", "example_submission"]:
            filename = f"{dataset}.{file_format}"
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
            params={"apiKey": self.apikey},
            files={"file": ("x", predictions.to_csv(index=False).encode('ascii'))},
        )

        if response.status_code == 200:
            logger.info("Submission submitted :)")
            submission_id = response.json()["id"]
            return submission_id
        elif response.status_code == 502:
            logger.error("The server is not available")
            logger.info("Please wait before retrying.")
        else:
            body = response.json()
            
            if "message" in body:
                logger.error(body["message"])
            elif "code" in body:
                logger.error(body["code"])
            else:
                logger.error(str(body))

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
