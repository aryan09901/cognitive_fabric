from fastapi import APIRouter
from knowledge.core.knowledge_graph import search_knowledge_graph

router = APIRouter()

@router.get("/knowledge/query")
def query_knowledge(query: str):
    return {"answer": search_knowledge_graph(query)}
