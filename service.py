from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from utils.search import Text2Img
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.schemas import SearchText

app = FastAPI()

app.mount("/images", StaticFiles(directory="images"), name="images")

templates = Jinja2Templates(directory='templates')

text2img = Text2Img()

@app.get("/", response_class=HTMLResponse)
async def show_search_form(request: Request):
    return templates.TemplateResponse(name="form_template.html", context={"request": request})

@app.post("/api/search", response_class=HTMLResponse)
async def create_item(request: Request):
    form_data = await request.form()
    
    try:
        form_data_dict = dict(form_data)
        search_text = SearchText(**form_data_dict)

        results = text2img.search(text=search_text.text)

        context = {
            'names': [res['path'].replace('\\', '/').split('/')[-1] for res in results],
            'request': request
        }

        return templates.TemplateResponse(request=request, name='images_template.html', context=context)

    except ValueError as e:
        context = {
            'request': request,
            'error': str(e)
        }
        return templates.TemplateResponse("form_template.html", context)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
