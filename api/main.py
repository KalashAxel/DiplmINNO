from src.app import get_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(get_app(), host="0.0.0.0", port=8000)
