from fastapi import FastAPI
from app import unsuperwhite_v11
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.unsuperwhite_v11.utils import PredictScore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.unsuperwhite_v11_model = PredictScore()
    yield
    del app.state.unsuperwhite_v11_model


app = FastAPI(lifespan=lifespan, docs_url=None)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/docs/swagger-ui-bundle.js",
        swagger_css_url="/static/docs/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


app.include_router(unsuperwhite_v11.router)
