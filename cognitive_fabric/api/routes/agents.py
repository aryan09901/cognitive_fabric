from fastapi import APIRouter

router = APIRouter()

@router.get("/agents")
def get_agents():
    return [{"name": "agent1"}, {"name": "agent2"}]
