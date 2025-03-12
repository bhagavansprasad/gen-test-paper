BASE_FOLDER="$(PWD)"
APP_FOLDER="$(BASE_FOLDER)/" 
LOGS_FOLDER=$(BASE_FOLDER)/logs
SERVICE="aignite"
PORT=8000
SHUTDOWN_WAIT_SECONDS=5

.PHONY: help start stop status restart clean logs utest e2e

help:
	@echo "Usage: make <command>"
	@echo ""
	@echo "Available commands:"
	@echo "  make start     - Start FastAPI server"
	@echo "  make stop      - Stop running server"
	@echo "  make restart   - Restart server"
	@echo "  make status    - Show the status of server"
	@echo "  make clean     - Clean logs and temporary files"
	@echo "  make showlogs  - Continuously display the server logs"
	@echo "  make utest     - Runs unit tests"
	@echo "  make e2e       - Runs end-to-end tests"
	@echo " "
	@echo "Example: make start"

start:
	@echo "Starting FastAPI server..."
	@mkdir -p "$(LOGS_FOLDER)"
	@echo "Starting $(SERVICE) on port $(PORT)..."
	
	@echo "Executing: nohup uvicorn app.main:app --host 0.0.0.0 --port $(PORT) \
	--reload --app-dir $(APP_FOLDER) > $(LOGS_FOLDER)/$(SERVICE).log 2>&1 &"
	
	@(nohup uvicorn app.main:app \
	--host 0.0.0.0 \
	--port $(PORT) \
	--reload \
	--app-dir $(APP_FOLDER) > "$(LOGS_FOLDER)/$(SERVICE).log" 2>&1 &)

	@echo "$(SERVICE) started. Logs: $(LOGS_FOLDER)/$(SERVICE).log"
	@echo "Server started successfully!"
	@echo "Waiting for server to initialize..."
	@echo "Executing: sleep 5 && head -n 20 $(LOGS_FOLDER)/$(SERVICE).log"
	
	@(sleep 5 && head -n 20 "$(LOGS_FOLDER)/$(SERVICE).log") || \
	(echo "Error: Log file not found or empty." && exit 1)

stop:
	@echo "Stopping server..."
	@pid=$$(lsof -ti :$(PORT) 2>/dev/null | head -n 1); \
	if [ -n "$$pid" ]; then \
		echo "Sending TERM signal to process on port $(PORT) (PID: $$pid)..."; \
		echo "Executing: kill -TERM $$pid"; \
		kill -TERM "$$pid" 2>/dev/null; \
		echo "Waiting for graceful shutdown ($(SHUTDOWN_WAIT_SECONDS) seconds)..."; \
		echo "Executing: sleep $(SHUTDOWN_WAIT_SECONDS)";\
		sleep $(SHUTDOWN_WAIT_SECONDS); \
		pid_after_wait=$$(lsof -ti :$(PORT) 2>/dev/null | head -n 1); \
		if [ -n "$$pid_after_wait" ]; then \
			echo "Process still running after wait. Sending KILL signal..."; \
			echo "Executing: kill -KILL $$pid_after_wait";\
			kill -KILL "$$pid_after_wait" 2>/dev/null; \
		else \
			echo "Process stopped gracefully."; \
		fi; \
	else \
		echo "No process running on port $(PORT)."; \
	fi
	@echo "Server stopped."

status:
	@echo "Checking server status..."
	@pid=$$(lsof -ti :$(PORT) 2>/dev/null | head -n 1); \
	if [ -n "$$pid" ]; then \
		echo "Server running on port $(PORT) (PID: $$pid)"; \
	else \
		echo "No process running on port $(PORT)"; \
	fi

showlogs:
	@echo "Tailing logs for $(SERVICE)... (Press Ctrl+C to stop)"
	@echo "Executing: tail -f $(LOGS_FOLDER)/$(SERVICE).log"
	@tail -f "$(LOGS_FOLDER)/$(SERVICE).log"

restart: stop start

clean:
	@echo "Cleaning log files and __pycache__ directories..."
	@echo "Executing: rm -rf $(LOGS_FOLDER)/*.log"
	@rm -rf "$(LOGS_FOLDER)"/*.log
	@echo "Executing: find $(BASE_FOLDER) -name __pycache__ -type d -exec rm -rf {} +"
	@find "$(BASE_FOLDER)" -name __pycache__ -type d -exec rm -rf {} +
	@echo "Cleaned."

utest:
	@echo "Running unit tests..."
	@pytest tests/unit/

e2e:
	@echo "Running end-to-end tests..."
	@pytest tests/e2e/
	
