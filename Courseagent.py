import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq


# --------------------------------------------------
# 1. Load Environment Variables
# --------------------------------------------------

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")


# --------------------------------------------------
# 2. Create Calculator Tool
# --------------------------------------------------

@tool
def calculator(a: float, b: float, operation: str) -> str:
    """
    Perform basic mathematical calculations.

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
        return "Unsupported operation. Use add, subtract, multiply, or divide."

    return f"Calculation result: {result}"


# --------------------------------------------------
# 3. Create Course Information Tool
# --------------------------------------------------

@tool
def get_course_info(course_name: str) -> str:
    """
    Get course information about Python, Java, Full Stack,
    Generative AI, or Data Science.

    Args:
        course_name: Name of the course.
    """

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
        available = ", ".join(courses.keys())
        return f"Course not found. Available courses: {available}"

    course = courses[course_name]

    return (
        f"Course: {course_name.title()}\n"
        f"Duration: {course['duration']}\n"
        f"Topics: {course['topics']}\n"
        f"Level: {course['level']}"
    )


# --------------------------------------------------
# 4. Create Programming Interview Question Tool
# --------------------------------------------------

@tool
def get_interview_questions(
    language: str,
    topic: str = "basics",
    difficulty: str = "beginner"
) -> str:
    """
    Get programming interview questions for a language and topic.

    Args:
        language: Programming language, such as Python, Java, or SQL.
        topic: Topic such as basics, OOP, or collections.
        difficulty: beginner, intermediate, or advanced.
    """

    question_bank = {
        "python": {
            "basics": [
                "What is Python?",
                "What are Python variables?",
                "What is the difference between a list and a tuple?",
                "What is the difference between == and is?",
                "What are Python data types?"
            ],
            "oop": [
                "What is a class and an object?",
                "What is inheritance?",
                "What is polymorphism?",
                "What is encapsulation?",
                "What is method overriding?"
            ],
            "collections": [
                "What is the difference between a list and a set?",
                "How does a dictionary work?",
                "What is a tuple?",
                "What is a list comprehension?",
                "How can you remove duplicates from a list?"
            ]
        },

        "java": {
            "basics": [
                "What is Java?",
                "What is JVM?",
                "What is the difference between JDK and JRE?",
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
                "What is the Java Collections Framework?",
                "What is the difference between ArrayList and LinkedList?",
                "What is a HashMap?",
                "What is the difference between List and Set?",
                "What is an Iterator?"
            ]
        },

        "sql": {
            "basics": [
                "What is SQL?",
                "What is a primary key?",
                "What is a foreign key?",
                "What is the difference between DELETE and TRUNCATE?",
                "What is the difference between WHERE and HAVING?"
            ],
            "joins": [
                "What is an INNER JOIN?",
                "What is a LEFT JOIN?",
                "What is a RIGHT JOIN?",
                "What is a FULL OUTER JOIN?",
                "What is a self join?"
            ]
        }
    }

    language = language.lower().strip()
    topic = topic.lower().strip()

    if language not in question_bank:
        return "Supported languages: Python, Java, and SQL."

    if topic not in question_bank[language]:
        available_topics = ", ".join(question_bank[language].keys())
        return f"Topic not found. Available topics for {language}: {available_topics}"

    questions = question_bank[language][topic]

    return (
        f"{difficulty.title()} interview questions for "
        f"{language.title()} - {topic.title()}:\n"
        + "\n".join(
            f"{index + 1}. {question}"
            for index, question in enumerate(questions)
        )
    )


# --------------------------------------------------
# 5. Initialize Groq Chat Model
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# --------------------------------------------------
# 6. Create the Agent
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

    Your responsibilities:

    1. Answer Python, Java, SQL, and programming questions.
    2. Use the calculator tool for mathematical calculations.
    3. Use the course information tool when students ask about courses.
    4. Use the interview question tool when students request
       programming interview questions.
    5. Automatically select the appropriate tool.
    6. Explain answers in simple English.
    7. If a question is not related to the available tools,
       answer using your general programming knowledge.
    8. Never invent course details. Use the course tool
       for information about the listed courses.
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
print("Ask programming questions, calculate values,")
print("explore courses, or practice interview questions.")
print("Type 'exit' to stop the application.")
print("------------------------------------------")


while True:

    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\nAssistant: Happy learning! Goodbye.")
        break

    if not user_input:
        continue

    try:
        # Add the new user question to previous conversation
        messages.append(HumanMessage(content=user_input))

        # Invoke the agent with conversation history
        response = agent.invoke({
            "messages": messages
        })

        # Save the complete updated conversation
        messages = response["messages"]

        # Get the latest assistant response
        assistant_response = messages[-1].content

        print("\nAssistant:", assistant_response)

    except Exception as error:
        print("\nError:", error)
        print("Please check your API key, model access, and internet connection.")