import openai
import base64
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up OpenAI client
client = openai.OpenAI()
model_id = 'gpt-4o-mini'

def evaluate_exam(image_path='filled_exam.png'):
    logger.info(f"Starting image processing for: {image_path}")
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            content = [
                {"type": "text", "text": "Analyze this filled exam sheet. Identify the questions, the given answers, and evaluate their correctness. Provide a detailed assessment of the exam performance."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
            
            logger.info("Sending request to OpenAI API")
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=600
            )
        
        logger.info("Successfully processed image")
        output = response.choices[0].message.content
        logger.info(f"Output: {output}")
        return output
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return None
    except Exception as e:
        logger.exception(f"Error processing image: {str(e)}")
        return None

# Test the function
if __name__ == "__main__":
    result = evaluate_exam()
    if result and "I'm sorry, but I can't analyze the contents of the image" not in result:
        logger.info("Function executed successfully")
        logger.info(f"Exam evaluation:\n{result}")
    else:
        logger.error("Function failed to execute properly or couldn't analyze the exam image")
