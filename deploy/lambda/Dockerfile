FROM public.ecr.aws/lambda/python:3.9

# Set working directory
WORKDIR /var/task

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Lambda handler and all source code
COPY handler.py .

# Copy the entire models directory, including weights and Python modules
COPY models/ models/

# Workaround: Copy agent_lw.py to /var/task/ for pickle compatibility if needed
COPY models/dqn/agent_lw.py /var/task/
COPY models/dqn/cooldown_manager.py /var/task/


# Set the CMD to your Lambda handler
CMD ["handler.lambda_handler"]  