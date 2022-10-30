# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_predict_custom_trained_model_sample]
from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.cloud import datastore
from google.oauth2 import service_account

# credentials = service_account.Credentials.from_service_account_file(
#     'service.json')
# credentials = Credentials(
#     token=None,
#     token_uri='https://oauth2.googleapis.com/token',
#     refresh_token='1//06Eo3HgV23KLpCgYIARAAGAYSNwF'
#                   '-L9Iri_LUmH9ctkO5z7yzbX9WQGAvjXrdHemimWWoyAPSYU3KW1YVdL70Nw2LzcS3KW2lOPc',
#     client_id='764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com',
#     client_secret='d-FL95Q19q7MQmFpd7hHD0Ty')
aiplatform.init(
    # your Google Cloud Project ID or number
    # environment default used if not set
    project='My First Project',

    # the Vertex AI region you will use
    # defaults to us-central1
    location='us-west2',

    # Google Cloud Storage bucket in same region as location
    # used to stage artifacts
    staging_bucket='gs://conversational-agent',

    # custom google.auth.credentials.Credentials
    # environment default creds used if not set
    credentials=credentials,

    # customer managed encryption key resource name
    # will be applied to all Vertex AI resources if set

    # the name of the experiment to use to track
    # logged metrics and parameters
    # experiment='my-experiment',
    #
    # # description of the experiment above
    # experiment_description='my experiment description'
)


def predict_custom_trained_model_sample(
        project: str,
        endpoint_id: str,
        instances: Union[Dict, List[Dict]],
        location: str = "us-central1",
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(credentials=credentials)
    # client = datastore.Client(project='My First Project',
    #                           # the Vertex AI region you will use
    #                           # defaults to us-central1
    #                           client_options=client_options,
    #                           # custom google.auth.credentials.Credentials
    #                           # environment default creds used if not set
    #                           credentials=credentials)
    # The format of each instance should conform to the deployed model's prediction input schema.
    # instances = instances if type(instances) == list else [instances]
    # instances = [
    #     json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    # ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

# [END aiplatform_predict_custom_trained_model_sample]
