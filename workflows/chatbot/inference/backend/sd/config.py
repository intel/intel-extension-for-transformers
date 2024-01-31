# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import socket

MODEL_PATH="runwayml/stable-diffusion-v1-5"
LOG_DIR="./running_log"
IP_FLAG="2"
SQL_HOST="database-sd.cnhq0zedh6yu.us-east-1.rds.amazonaws.com"
SQL_PORT=3306
SQL_USER="admin"
SQL_PASSWORD=""
SQL_DB="db_sd"
SQL_TABLE="test"
SERVER_HOST=socket.gethostname()
SERVER_PORT=80
