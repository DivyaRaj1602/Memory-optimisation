"""Entry point for multi-mode deployment."""

import uvicorn
from server.main import app


def main():
    uvicorn.run("server.main:app", host="0.0.0.0", port=8004, reload=False)


if __name__ == "__main__":
    main()
