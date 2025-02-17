from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from src.exception import CustomException
from src.config import Config
from src.data_ingestion import ingest_files
from src.file_loader import File
from typing import List, TypedDict, Iterable
from enum import Enum
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, Document, AIMessage, HumanMessage
# from langchain.llms import ChatOllama
from langchain_ollama import ChatOllama
import sys

SYSTEM_PROMPT = """
You're having a conversation with an user about excerpts of their files. Try to be helpful and answer their questions.

If you don't know the answer, say that you don't know and try to ask clarifying questions.
""".strip()

PROMPT = """
Here's the information you have about the excerpts of the files:

<context>
{context}
</context>

One file can have multiple excerpts.

Please, respond to the query below:

<question>
{question}
</question>

Answer:

"""

FILE_TEMPLATE="""
<file>
<name>{name}</name>
<content>{content}</content>
</file>

""".strip()

PROMPT_TEMPLATE=ChatPromptTemplate.from_messages(

    [
        (
            "system",
            SYSTEM_PROMPT,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",PROMPT)
    ]

)

class Role(Enum):
    USER="user"
    ASSISTANT="assistant"

@dataclass
class Message:
    role: Role 
    content: str

    

@dataclass
class ChunkEvent:
    content: str

@dataclass
class SourcesEvent:
    content: List[Document]

@dataclass
class FinalAnswerEvent:
    content: str

class State(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    context: List[Document]
    answer:str
    
def _remove_thinking_from_message(message: str) -> str:
    close_tag = "</think>"
    tag_length = len(close_tag)
    return message[message.find(close_tag) + tag_length :].strip()

def create_history(welcome_message: Message) -> List[Message]:
    return [welcome_message]


class Chatbot:
    def __init__(self, files: List[File]):
        self.files = files
        self.retriever = ingest_files(files)
        self.llm = ChatOllama(
            model=Config.Model.NAME,
            temperature=Config.Model.TEMPERATURE,
            verbose=False,
            keep_alive=-1,
        )
        self.workflow = self._create_workflow()

    def _format_docs(self, docs: List[Document]) -> str:
        try:
            return "\n\n".join(
                FILE_TEMPLATE.format(name=doc.metadata["source"], content=doc.page_content)
                for doc in docs
            )
        
        except Exception as e:
            raise CustomException(e,sys)

    def _retrieve(self, state: State):
        try:
            context = self.retriever.invoke(state["question"])
            return {"context": context}
        except Exception as e:
            raise CustomException(e,sys)
    
    def _generate(self, state: State):
        try:
            messages = PROMPT_TEMPLATE.invoke(
                {
                    "question": state["question"],
                    "context": self._format_docs(state["context"]),
                    "chat_history": state["chat_history"],
                }
            )
            answer = self.llm.invoke(messages)
            return {"answer": answer}
        except Exception as e:
            raise CustomException(e,sys)

    def _create_workflow(self) -> CompiledStateGraph:
        graph_builder = StateGraph(State).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        return graph_builder.compile()
    
    def _ask_model(
        self, prompt: str, chat_history: List[Message]
    ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        try:
            history = [
                AIMessage(m.content) if m.role == Role.ASSISTANT else HumanMessage(m.content)
                for m in chat_history
            ]
            payload = {"question": prompt, "chat_history": history}

            config = {
                "configurable": {"thread_id": 42},
            }
            for event_type, event_data in self.workflow.stream(
                payload,
                config=config,
                stream_mode=["updates", "messages"],
            ):
                if event_type == "messages":
                    chunk, _ = event_data
                    yield ChunkEvent(chunk.content)

                if event_type == "updates":
                    if "_retrieve" in event_data:
                        documents = event_data["_retrieve"]["context"]
                        yield SourcesEvent(documents)

                    if "_generate" in event_data:
                        answer = event_data["_generate"]["answer"]
                        yield FinalAnswerEvent(answer.content)
        except Exception as e:
            raise CustomException(e,sys)
                

    def ask(
        self, prompt: str, chat_history: List[Message]
    ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        try:
            for event in self._ask_model(prompt, chat_history):
                yield event
                if isinstance(event, FinalAnswerEvent):
                    response = _remove_thinking_from_message("".join(event.content))
                    chat_history.append(Message(role=Role.USER, content=prompt))
                    chat_history.append(Message(role=Role.ASSISTANT, content=response))
        except Exception as e:
            raise CustomException(e,sys)