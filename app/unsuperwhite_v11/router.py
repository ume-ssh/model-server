from fastapi import APIRouter, Depends, Body, Request
from .schemas import UnsuperwhiteV11Data, ScoreResponse
from .utils import PredictScore
from .payload_example import unsuperwhite_v11_example


router = APIRouter()

predict_score_instance = None


def get_model(request: Request) -> PredictScore:
    return request.app.state.unsuperwhite_v11_model


@router.post("/score")
def score(
    data: UnsuperwhiteV11Data = Body(example=unsuperwhite_v11_example),
    predict_score_instance: PredictScore = Depends(get_model),
) -> ScoreResponse:
    result = predict_score_instance.get_score(data.dict())
    return result
