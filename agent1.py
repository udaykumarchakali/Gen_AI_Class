import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


# --------------------------------------------------
# 1. Load Environment Variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. Add it to your .env file."
    )


# --------------------------------------------------
# 2. Create Calculator Tool
# --------------------------------------------------

@tool
def calculator(a: float, b: float, operation: str) -> str:
    """Perform basic mathematical calculations.

    Args:
        a: First number.
        b: Second number.
        operation: add, subtract, multiply, or divide.
    """

    operation = operation.lower().strip()

    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero is not allowed."
        result = a / b
    else:
        return "Unsupported operation."

    return f"Calculation result: {result}"


# 
# 3. Create Course Information Tool
#  

@tool
def get_course_info(course_name: str) -> str:
    """Get information about available programming courses."""

    courses = {
        "python": {
            "duration": "3 months",
            "topics": "Python basics, OOP, MySQL, Django, FastAPI",
            "level": "Beginner to Advanced"
        },
        "java": {
            "duration": "4 months",
            "topics": "Core Java, OOP, JDBC, Spring Boot, Microservices",
            "level": "Beginner to Advanced"
        },
        "full stack": {
            "duration": "5 months",
            "topics": "HTML, CSS, JavaScript, React, Python/Java, APIs, Database",
            "level": "Beginner to Advanced"
        },
        "generative ai": {
            "duration": "4 months",
            "topics": "Python, LLMs, Prompt Engineering, RAG, LangChain, AI Agents",
            "level": "Beginner to Advanced"
        },
        "data science": {
            "duration": "4 months",
            "topics": "Python, NumPy, Pandas, Statistics, Machine Learning",
            "level": "Beginner to Advanced"
        }
    }

    course_name = course_name.lower().strip()

    if course_name not in courses:
        return "Available courses: " + ", ".join(courses.keys())

    course = courses[course_name]

    return (
        f"Course: {course_name.title()}\n"
        f"Duration: {course['duration']}\n"
        f"Topics: {course['topics']}\n"
        f"Level: {course['level']}"
    )


# --------------------------------------------------
# 4. Create Interview Questions Tool
# --------------------------------------------------

@tool
def get_interview_questions(
    language: str,
    topic: str = "basics",
    difficulty: str = "beginner"
) -> str:
    """Get programming interview questions.

    Args:
        language: Python, Java, or SQL.
        topic: basics, OOP, collections, or joins.
        difficulty: beginner, intermediate, or advanced.
    """

    question_bank = {
        "python": {
            "basics": [
                "What is Python?",
                "What are Python variables?",
                "Difference between list and tuple?",
                "Difference between == and is?",
                "What are Python data types?"
            ],
            "oop": [
                "What is a class and object?",
                "What is inheritance?",
                "What is polymorphism?",
                "What is encapsulation?",
                "What is method overriding?"
            ],
            "collections": [
                "Difference between list and set?",
                "How does a dictionary work?",
                "What is a tuple?",
                "What is list comprehension?",
                "How to remove duplicates from a list?"
            ]
        },
        "java": {
            "basics": [
                "What is Java?",
                "What is JVM?",
                "Difference between JDK and JRE?",
                "What are Java data types?",
                "Why is Java platform independent?"
            ],
            "oop": [
                "What is inheritance in Java?",
                "What is method overloading?",
                "What is method overriding?",
                "What is an abstract class?",
                "What is an interface?"
            ],
            "collections": [
                "What is Java Collections Framework?",
                "Difference between ArrayList and LinkedList?",
                "What is HashMap?",
                "Difference between List and Set?",
                "What is an Iterator?"
            ]
        },
        "sql": {
            "basics": [
                "What is SQL?",
                "What is a primary key?",
                "What is a foreign key?",
                "Difference between DELETE and TRUNCATE?",
                "Difference between WHERE and HAVING?"
            ],
            "joins": [
                "What is INNER JOIN?",
                "What is LEFT JOIN?",
                "What is RIGHT JOIN?",
                "What is FULL OUTER JOIN?",
                "What is a self join?"
            ]
        }
    }

    language = language.lower().strip()
    topic = topic.lower().strip()

    if language not in question_bank:
        return "Supported languages: Python, Java, and SQL."

    if topic not in question_bank[language]:
        available = ", ".join(question_bank[language].keys())
        return f"Available topics: {available}"

    questions = question_bank[language][topic]

    return (
        f"{difficulty.title()} interview questions for "
        f"{language.title()} - {topic.title()}:\n"
        + "\n".join(
            f"{i + 1}. {q}"
            for i, q in enumerate(questions)
        )
    )


# --------------------------------------------------
# 5. Initialize OpenRouter Model
# --------------------------------------------------

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


# --------------------------------------------------
# 6. Create Agent
# --------------------------------------------------

tools = [
    calculator,
    get_course_info,
    get_interview_questions
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are a Personal Learning Assistant for programming students.

    Responsibilities:

    1. Answer Python, Java, SQL, and programming questions.
    2. Use calculator for mathematical calculations.
    3. Use course information tool for course-related questions.
    4. Use interview question tool for interview preparation.
    5. Automatically select the appropriate tool.
    6. Explain answers in simple English.
    7. Use general programming knowledge for other questions.
    8. Never invent course details.
    """
)


# --------------------------------------------------
# 7. Maintain Conversation History
# --------------------------------------------------

messages = []


# --------------------------------------------------
# 8. Start Chat Application
# --------------------------------------------------

print("\n==========================================")
print("   PERSONAL LEARNING ASSISTANT AGENT")
print("==========================================")

print("Powered by OpenRouter + LangChain")
print("Type exit to stop the application.")

print("------------------------------------------")


while True:

    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\nAssistant: Happy learning! Goodbye.")
        break

    if not user_input:
        continue

    try:

        messages.append(
            HumanMessage(content=user_input)
        )

        response = agent.invoke({
            "messages": messages
        })

        messages = response["messages"]

        assistant_response = messages[-1].content

        print("\nAssistant:", assistant_response)

    except Exception as error:

        print("\nError:", error)
        print("Check API key, model availability, and internet connection.")