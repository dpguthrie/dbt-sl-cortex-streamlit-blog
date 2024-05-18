e/* Grant database role snowflake.cortex_user */

use role accountadmin;

create role cortex_user_role;
grant database role snowflake.cortex_user to role cortex_user_role;

grant role cortex_user_role to user some_user;

/* Creating Streamlit Apps */

-- If you want all roles to create Streamlit apps, run
grant usage on database <database_name> to role public;
grant usage on schema <database_name>.<schema_name> to role public;
grant create streamlit on schema <database_name>.<schema_name> to role public;
grant create stage on schema <database_name>.<schema_name> to role public;

-- Don't forget to grant USAGE on a warehouse (if you can).
grant usage on warehouse <warehouse_name> to role public;

-- If you only want certain roles to create Streamlit apps, 
-- change the role name in the above commands.

/* Network Rules, External Access Integration, Secret */

grant create network rule on schema <database_name>.<schema_name> to role public;
grant create secret on schema <database_name>.<schema_name> to role public;

use database <database_name>;
use schema <schema_name>;

create or replace network rule dbt_cloud_semantic_layer_rule
    mode = egress
    type = host_port
    value_list = (
        'semantic-layer.cloud.getdbt.com',
        'semantic-layer.emea.dbt.com',
        'semantic-layer.au.dbt.com'
    );

create or replace secret dbt_cloud_service_token
    type = generic_string
    secret_string = '<service_token>';

create or replace external access integration dbt_cloud_semantic_layer_integration
    allowed_network_rules = (dbt_cloud_semantic_layer_rule)
    allowed_authentication_secrets = (dbt_cloud_service_token)
    enabled = true;

grant usage on integration dbt_cloud_semantic_layer_integration to role public;
grant ownership on secret dbt_cloud_service_token to role public;
grant usage on secret dbt_cloud_service_token to role public;

/* UDF - Retrieve Initial Metadata */

create or replace function retrieve_sl_metadata(host string, environment_id integer, token string default null)
    returns object
    language python
    runtime_version = 3.9
    handler = 'main'
    external_access_integrations = (dbt_cloud_semantic_layer_integration)
    packages = ('requests')
    secrets = ('cred' = dbt_cloud_service_token)
as
$$
from typing import Dict
import _snowflake
import requests

query = """
query GetMetrics($environmentId: BigInt!) {
  metrics(environmentId: $environmentId) {
    description
    name
    queryableGranularities
    type
    dimensions {
      description
      name
      type
    }
  }
}
"""

def main(host: str, environment_id: int, token: str = None):
    session = requests.Session()
    if host[-1] == '/':
        host = host[:-1]
    url = f'https://{host}/api/graphql'
    if token == "":
        token = _snowflake.get_generic_secret_string('cred')
    session.headers = {'Authorization': f'Bearer {token}'}
    payload = {"query": query, "variables": {"environmentId": environment_id}}
    response = session.post(url, json=payload)
    response.raise_for_status()
    return response.json()

$$;

grant usage on function retrieve_sl_metadata() to role public;

/* UDF - Submit SL Request, Return Data */

create or replace function submit_sl_request(payload string)
    returns object
    language python
    runtime_version = 3.9
    handler = 'main'
    external_access_integrations = (dbt_cloud_semantic_layer_integration)
    packages = ('requests')
    secrets = ('cred' = dbt_cloud_service_token )
as
$$
from typing import Dict
import _snowflake
import json
import requests


def main(payload: str):
    session = requests.Session()
    token = _snowflake.get_generic_secret_string('cred')
    session.headers = {'Authorization': f'Bearer {token}'}
    payload = json.loads(payload)
    results = submit_request(session, payload)
    try:
        query_id = results["data"]["createQuery"]["queryId"]
    except TypeError as e:
        raise e
    
    data = None
    while True:
        graphql_query = """
            query GetResults($environmentId: BigInt!, $queryId: String!) {
                query(environmentId: $environmentId, queryId: $queryId) {
                    arrowResult
                    error
                    queryId
                    sql
                    status
                }
            }
        """
        results_payload = {"variables": {"queryId": query_id}, "query": graphql_query}
        results = submit_request(session, results_payload)
        try:
            data = results["data"]["query"]
        except TypeError as e:
            break
        else:
            status = data["status"].lower()
            if status in ["successful", "failed"]:
                break

    return data

def submit_request(session: requests.Session, payload: Dict):
    if not "variables" in payload:
        payload["variables"] = {}
    payload["variables"].update({"environmentId": 1})
    response = session.post(
        "https://semantic-layer.cloud.getdbt.com/api/graphql", json=payload
    )
    response.raise_for_status()
    return response.json()
$$;

grant usage on function submit_sl_request(string) to role public;
