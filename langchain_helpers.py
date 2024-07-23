import re
import json
import requests
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Type

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import ConfigurableField, ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain_core.chat_history import BaseChatMessageHistory
#from langchain_community.chat_message_histories import ChatMessageHistory, CosmosDBChatMessageHistory
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

#custom libraries that we will use later in the app
#from common.utils import  GetDocSearchResults_Tool
#from common.prompts import AGENT_DOCSEARCH_PROMPT

from collections import OrderedDict

from backend.settings import (
    app_settings,
    MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
)

#custom libraries that we will use later in the app
#from common.utils import  GetDocSearchResults_Tool
#from common.prompts import AGENT_DOCSEARCH_PROMPT

from collections import OrderedDict

## Credenciales
#piecewise = False
#AISEARCH
search_api_version= app_settings.azure_openai.preview_api_version
search_query_type = app_settings.datasource.query_type
search_service_endpoint = app_settings.datasource.endpoint
search_service_key = app_settings.datasource.key

#OPENAI
openai_api_key = app_settings.azure_openai.key
openai_api_base = app_settings.azure_openai.endpoint
openai_api_version = app_settings.azure_openai.preview_api_version
openai_embeddings_model = app_settings.azure_openai.model
openai_temperature = app_settings.azure_openai.temperature
## RAG MODEL - AGENT
def get_search_results(query: str, indexes: list, 
                       k: int = 5,
                       score_threshold: float = 0.78,
                       rel_threshold: float = 0.15,
                       sas_token: str = "",
                       verbose: bool = False) -> List[dict]:
    
    """Performs multi-index hybrid search and retu,rns ordered dictionary with the combined results"""
    
    #credential = DefaultAzureCredential()
    #token = credential.get_token("https://search.azure.com/.default")
    #access_token = token.token
    headers = {'Content-Type': 'application/json','api-key': search_service_key 
               #,'Authorization': f"Bearer {access_token}"
              }
    params = {'api-version': search_api_version}
    
    agg_search_results = dict()
    
    for index in indexes:
        search_payload = {
            "search": "*",
            "select": "filepath, id, contentVector, title, name, content, url", #se realiza así la búsqueda?
            #"queryType": "vector",
            "vectorQueries": [{"text": query, "fields": "contentVector", "kind": "text", "k": k}],
            #"semanticConfiguration": "my-semantic-config",
            #"captions": "extractive",
            #"answers": "extractive",
            "count":"true",
            "top": k    
        }

        resp = requests.post(search_service_endpoint + "/indexes/" + index +"/docs/search",
                     data=json.dumps(search_payload), headers=headers, params=params)

        search_results = resp.json()
        agg_search_results[index] = search_results
    
    content = dict()
    ordered_content = OrderedDict()
    
    
    for index,search_results in agg_search_results.items():
        for result in search_results['value']:
            if result['@search.score'] > score_threshold: # Show results that are at least N% of the max possible score=4
                content[result['id']]={
                                        "title": result['title'].replace("\n",""), 
                                        "name": result['name'], 
                                        "content": re.sub("[\n\s*][\n\s*][\n\s*]+","\n\n",result['content']),
                                        "score": result['@search.score'],
                                        "index": index,
                                        "location": result["filepath"],
                                        "url": result["url"]
                                    }
                

    topk = k
    count = 0  # To keep track of the number of results added
    sorted_ids = sorted(content, key=lambda x: content[x]["score"], reverse=True)
    if not sorted_ids:
        return []
    max_score = content[sorted_ids[0]]["score"]

    for id in sorted_ids:
        ordered_content[id] = content[id]
        count += 1
        if count >= topk or (max_score - content[id]["score"]) > rel_threshold:  # Stop after adding topK results
            break
    
    if verbose:
        for v in ordered_content.values():
            print(f"{v['title']}: {v['score']}")

    final_l = list(ordered_content.values())
    return final_l

class GetDocSearchResults(BaseRetriever):
    indexes: List[str] = []
    k: int = 10
    score_th: float = 0.79
    sas_token: str = ""
    rel_th : float = 0.15

    def _get_relevant_documents(self, query, *, runManager = None):
        return get_search_results(query = query, indexes = self.indexes, 
                       k = self.k,
                       score_threshold = self.score_th,
                       rel_threshold = self.rel_th,
                       sas_token = self.sas_token,
                       verbose = False)
    async def _aget_relevant_documents(self, query, *, runManager = None):
        return get_search_results(query = query, indexes = self.indexes, 
                       k = self.k,
                       score_threshold = self.score_th,
                       rel_threshold = self.rel_th,
                       sas_token = self.sas_token,
                       verbose = False)


#get gpt4
#Se establecen las herramientas a utilizar. 
#Tools are functions that an agent can invoke. If you don't give the agent access to a correct set of tools,
#it will never be able to accomplish the objectives you give it. 
#If you don't describe the tools well, the agent won't know how to use them properly.

blob_sas_token=""
index_format = "{:" + app_settings.custom.index_format + "}"
indexes=[app_settings.custom.index_root + index_format.format(i) for i in range(1,app_settings.custom.index_number + 1)]

#se establecen las herramientas que el agente utilizará
retriever = GetDocSearchResults(indexes=indexes, k=app_settings.datasource.top_k, score_th=app_settings.custom.absolute_threshold, rel_th=app_settings.custom.relative_threshold, sas_token=blob_sas_token)
#Definición del LLM a utilizar y vinculación a las herramientas.
#Los modelos pueden llamar a varias herramientas/funciones. Se pueden llamar a varias funciones simultaneamente. 
#El modelo elige que función utilizar. Elegirá que indice se utiliza.

from langchain_core.pydantic_v1 import BaseModel, Field


class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
''' 
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
'''
gpt_deployment_name=app_settings.azure_openai.model

COMPLETION_TOKENS = 1500
llm = AzureChatOpenAI(deployment_name=gpt_deployment_name, 
                      api_version=openai_api_version,
                      api_key=openai_api_key,
                      azure_endpoint=openai_api_base,
                      #azure_ad_token_provider=token_provider, #MANAGED IDENITTY
                      temperature=openai_temperature,
                      max_tokens=COMPLETION_TOKENS, streaming=True)

structured_llm = llm.with_structured_output(CitedAnswer, include_raw=True)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

roles = {"system":"system",
        "user":"human",
        "assistant":"ai"}

from langchain_core.runnables import RunnablePassthrough

def formatDocs(docs):
    return "\n**********************************\n".join(f"ID: {i+1}\nUrl: {doc['url']}\nContents: {doc['content']}" for i,doc in enumerate(docs))

def create_chain(messages):
    system_template = app_settings.azure_openai.system_message + "\nCuando en medio del texto quieras poner una referencia (citation) pon: [ID] reemplazando el ID por el de el documento." + "\nAquí están los documentos relevantes:\n{context}"
    user_template = "{input}"
    raw_prompt = [("system", system_template)] + \
                 [(roles[message.get("role")], message.get("content")) for message in messages if message.get("role") in roles] + \
                 [("human", user_template)]
    prompt = ChatPromptTemplate.from_messages(raw_prompt)
    
    retrieve_docs = (lambda x: x["input"]) | retriever
    pre_chain = RunnablePassthrough.assign(context = lambda x: formatDocs(x["context"])) | prompt | structured_llm
    return RunnablePassthrough.assign(context = retrieve_docs).assign(answer = pre_chain)

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import time

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import time

def format_context(cont, id):
    choices = [Choice(index = 0, finish_reason=None, delta = ChoiceDelta(content = None, role="assistant", context = {"citations": cont, "intent": '[]'}))]
    return ChatCompletionChunk(id = id, choices = choices, model=gpt_deployment_name, created=int(time.time()), object = "chat.completion.chunk")
async def convert_to_ChatCompletion(lc_answer):
    id = lc_answer["answer"]["raw"].id
    yield format_context(lc_answer["context"], id)
    lc = lc_answer["answer"]["parsed"]
    choices = [Choice(index = 0, finish_reason=None, delta = ChoiceDelta(content = re.sub(r"\[(\d+)\]",r"[doc\g<1>]",lc.answer), role="assistant"))]
    yield ChatCompletionChunk(id = id, choices = choices, model=gpt_deployment_name, created=int(time.time()), object = "chat.completion.chunk")

    for i in lc.citations:
        choices = [Choice(index = 0, finish_reason=None, delta = ChoiceDelta(content = f"[doc{i}]", role=None))]
        yield ChatCompletionChunk(id = id, choices = choices, model=gpt_deployment_name, created=int(time.time()), object = "chat.completion.chunk")
    yield ChatCompletionChunk(id = id, choices = [Choice(index=0, finish_reason="stop", delta = ChoiceDelta())], model = gpt_deployment_name, created=int(time.time()), object = "chat.completion.chunk")


async def stream_completions(chain,question):
    async for x in convert_to_ChatCompletion(chain.invoke(dict(input=question))):
        yield x