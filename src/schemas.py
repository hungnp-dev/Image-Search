from pydantic import BaseModel, field_validator

class SearchText(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def check_text(cls, v: str) -> str:
        res = v.strip() and all(c.isalpha() or c.isspace() for c in v)
        
        if not res:
            raise ValueError('Văn bản tìm kiếm không được rỗng và chỉ được chứa chữ cái và khoảng trắng')
        
        return v
