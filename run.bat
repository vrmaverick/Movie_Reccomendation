@echo off
docker build -t movie-recommender .
docker run -d -p 8080:80 --name movie-recommender-container movie-recommender
timeout /t 3 /nobreak
start http://localhost:8080