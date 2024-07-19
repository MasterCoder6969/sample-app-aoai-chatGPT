import asyncio
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
piecewise = True

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
                       score_threshold: float = 0.8,
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
            "select": "filepath, id, contentVector, title, name, content", #se realiza así la búsqueda?
            "queryType": "vector",
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
                                        "title": result['title'], 
                                        "name": result['name'], 
                                        "content": result['content'],
                                        "score": result['@search.score'],
                                        "index": index,
                                        "location": result["filepath"]
                                    }
                

    topk = k
    count = 0  # To keep track of the number of results added
    sorted_ids = sorted(content, key=lambda x: content[x]["score"], reverse=True)
    max_score = content[sorted_ids[0]]["score"]

    for id in sorted_ids:
        ordered_content[id] = content[id]
        count += 1
        if count >= topk or (max_score - content[id]["score"]) > rel_threshold:  # Stop after adding topK results
            break
    
    if verbose:
        for v in ordered_content.values():
            print(f"{v['title']}: {v['score']}")

    return ordered_content

class CustomAzureSearchRetriever(BaseRetriever):
    
    indexes: List
    topK : int
    score_threshold : float
    sas_token : str = ""
    rel_threshold : float
    
    
    def _get_relevant_documents(
        self, input: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        ordered_results = get_search_results(input, self.indexes, k=self.topK, score_threshold=self.score_threshold, rel_threshold=self.rel_threshold, sas_token=self.sas_token)
        
        top_docs = []
        for key,value in ordered_results.items():
            
            #esto creo que hay que eliminarlo
            location = value["location"] if value["location"] is not None else ""
            top_docs.append(Document(page_content=value["content"], metadata={"source": location, "score":value["score"]}))

        return top_docs

    
class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")
    return_direct: bool = Field(
        description="Whether or the result of this should be returned directly to the user without you seeing what it is",
        default=False,
    )
class GetDocSearchResults_Tool(BaseTool):
    name = "docsearch"
    description = "useful when the questions includes the term: docsearch"
    args_schema: Type[BaseModel] = SearchInput
    
    indexes: List[str] = []
    k: int = 10
    score_th: float = 0.79
    sas_token: str = ""
    rel_th : float = 0.15

    def _run(
        self, query: str,  return_direct = False, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:

        retriever = CustomAzureSearchRetriever(indexes=self.indexes, topK=self.k, score_threshold=self.score_th, rel_threshold=self.rel_th,
                                               sas_token=self.sas_token, callback_manager=self.callbacks)
        results = retriever.invoke(input=query)
        
        return results

    async def _arun(
        self, query: str, return_direct = False, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        
        retriever = CustomAzureSearchRetriever(indexes=self.indexes, topK=self.k, score_threshold=self.score_th, rel_threshold=self.rel_th, 
                                               sas_token=self.sas_token, callback_manager=self.callbacks)
        # Please note below that running a non-async function like run_agent in a separate thread won't make it truly asynchronous. 
        # It allows the function to be called without blocking the event loop, but it may still have synchronous behavior internally.
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(ThreadPoolExecutor(), retriever.invoke, query)
        
        return results

#get gpt4
#Se establecen las herramientas a utilizar. 
#Tools are functions that an agent can invoke. If you don't give the agent access to a correct set of tools,
#it will never be able to accomplish the objectives you give it. 
#If you don't describe the tools well, the agent won't know how to use them properly.

blob_sas_token=""
index_format = "{:" + app_settings.custom.index_format + "}"
indexes=[app_settings.custom.index_root + index_format.format(i) for i in range(1,app_settings.custom.index_number)]

#se establecen las herramientas que el agente utilizará
tools = [GetDocSearchResults_Tool(indexes=indexes, k=app_settings.datasource.top_k, score_th=app_settings.custom.absolute_threshold, rel_th=app_settings.custom.relative_threshold, sas_token=blob_sas_token)] #tengo mis dudas del uso del sas!
#Definición del LLM a utilizar y vinculación a las herramientas.
#Los modelos pueden llamar a varias herramientas/funciones. Se pueden llamar a varias funciones simultaneamente. 
#El modelo elige que función utilizar. Elegirá que indice se utiliza.

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
''' 
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
'''

gpt_deployment_name="model-gpt4"

COMPLETION_TOKENS = 1500
llm = AzureChatOpenAI(deployment_name=gpt_deployment_name, 
                      api_version=openai_api_version,
                      api_key=openai_api_key,
                      azure_endpoint=openai_api_base,
                      #azure_ad_token_provider=token_provider, #MANAGED IDENITTY
                      temperature=openai_temperature,
                      max_tokens=COMPLETION_TOKENS, streaming=True)

llm_with_tools = llm.bind_tools(tools) #se vinculan estas herramientas o funciones a las llamadas del modelo.
                                        #Cada vez qu ese invoque al mokdelo, se tendrá acceso a estas herramientas

roles = {"system":"system",
        "user":"human",
        "assistant":"ai"}

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import time

async def convert_to_ChatCompletion(lc_answer):
    index = 0
    async for lc in lc_answer:
        choices = [Choice(index = index, finish_reason=None, delta = ChoiceDelta(content = lc.content, role="assistant"))]
        index += 1
        yield ChatCompletionChunk(id = lc.id, choices = choices, model=gpt_deployment_name, created=int(time.time()), object = "chat.completion.chunk")
            
async def stream_completions(llm,prompt,question):
    async for x in convert_to_ChatCompletion(llm.astream(prompt.invoke(question))):
        yield x