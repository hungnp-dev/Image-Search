from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from utils.search import Text2Img
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.schemas import SearchText
from utils.path_utils import join_paths, ensure_dir, convert_path_for_url, get_file_name
import os

app = FastAPI()

# Sử dụng đường dẫn tương đối từ thư mục hiện tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = ensure_dir(join_paths(BASE_DIR, "images"))
TEMPLATES_DIR = join_paths(BASE_DIR, "templates")

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

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

        # Xử lý đường dẫn cho URL
        context = {
            'names': [get_file_name(convert_path_for_url(res['path'])) for res in results],
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
