
#%%
import json
from aind_data_access_api.document_db import MetadataDbClient

API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"

docdb_api_client = MetadataDbClient(
    host=API_GATEWAY_HOST,
    database=DATABASE,
    collection=COLLECTION,
)

filter = {"subject.subject_id": "746346"}
count = docdb_api_client._count_records(
    filter_query=filter,
)
print(count)

#%%
import json
from aind_data_access_api.document_db import MetadataDbClient

API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"

docdb_api_client = MetadataDbClient(
    host=API_GATEWAY_HOST,
    database=DATABASE,
    collection=COLLECTION,
)

# To get the complete record, don't use a projection
# Projections are useful if you are only interested in a
# subset of data. It makes it faster to retrieve from the db
projection = {"_id": 1, "quality_control": 1}
filter = {"_id": "ffe10f3d-a15f-4ed1-b108-1eed4c23b48a"}
record = docdb_api_client.retrieve_docdb_records(
    projection=projection,
    filter_query=filter,
)
record[0]
# {'_id': 'ffe10f3d-a15f-4ed1-b108-1eed4c23b48a', 'quality_control': None}
# %%
