import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool, ScrapeElementFromWebsiteTool
from llama_index.core.llms import ChatMessage
from llama_index.core import ChatPromptTemplate
from llama_index.llms.openai import OpenAI
from textwrap import dedent
from pydantic import BaseModel
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.llms.gemini import Gemini
# from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()

def extract_information(transcript):
    llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv('OPENAI_API_KEY'))
    prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for summarizing and extracting insights transcripts.\n"
                "Generate a valid JSON in the following format:\n"
                "{json_example}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Here is the transcript: \n"
                "------\n"
                "{transcript}\n"
                "------"
            ),
        ),
    ])

    supplier_details = {
    "scholarship_details": [
        {
            "name": "str",
            "country": "str",
            "amount": "str",
            "deadline": "str",
            "Scholarship_form":"str",
            "Dates": "str",
            "eligibility": "str",
            "field_of_study": "str",
            "degree_level": "str",
            "duration": "str",
            "website": "str",
            "funding_type": "str",
            "application_process": "str",
            "is_legitimate": "str",
            "is_current": "str",
            "organization_verified": "str",
            "application_process_verified": "str",
            "potential_red_flags": "str",
            "verification_methods_used": "str",
            "last_verified_date": "str",
            "overall_trust_score": "int",
            "verification_notes": "str"
        }
    ]
}
    json_example = json.dumps(supplier_details)
    messages = prompt.format_messages(json_example=json_example, transcript=transcript)
    output = llm.chat(messages, response_format={"type": "json_object"}).message.content
    return output



def output(prompt):
    os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')
    os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
    os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
    os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)

    search_tool = SerperDevTool()
    web_rag_tool = WebsiteSearchTool()
    get_content_tool = ScrapeWebsiteTool()
    llm = ChatGoogleGenerativeAI(model="gemini-pro")


    scholarship_data = "https://scholarshipforme.com/, https://www.buddy4study.com/, https://scholarships.gov.in/, https://www.education.gov.in/scholarships"

    ScholarshipFinder_agent = Agent(
        role='International Scholarship Navigator',
        goal=f'Provide comprehensive, accurate, and tailored scholarship information to students seeking financial aid for their studies. Thoroughly analyze and extract relevant data from the following trusted sources: {scholarship_data}. For each inquiry, conduct an in-depth exploration of these websites, including PDFs and related links, to ensure the most up-to-date and pertinent information is provided.',
        backstory="""As a dedicated International Scholarship Navigator, I have spent over a decade helping students worldwide achieve their academic dreams through financial aid opportunities. My journey began as a financial aid counselor at a prestigious university, where I witnessed firsthand the transformative power of scholarships. This experience ignited my passion for democratizing access to education.

        Over the years, I've cultivated an extensive network of contacts in educational institutions, government agencies, and non-profit organizations across the globe. My expertise spans a wide range of scholarship types, from merit-based awards to need-based grants, covering various academic disciplines and degree levels.

        I've personally assisted thousands of students in securing millions of dollars in scholarship funding, navigating complex application processes, and uncovering hidden opportunities. My approach combines meticulous research, data analysis, and a deep understanding of individual student needs to create tailored scholarship strategies.

        In recent years, I've leveraged my expertise to develop advanced tools and methodologies for scholarship verification, ensuring that students receive only legitimate and current opportunities. My commitment to ethical practices and thorough verification has made me a trusted advisor in the field of international education financing.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[search_tool, web_rag_tool, get_content_tool],
        async_execution=True
    )

    ScholarshipFinder_task = Task(
    description="""As an International Scholarship Navigator, my primary responsibility is to identify, verify, and present the most suitable scholarship opportunities for students seeking financial aid for their studies. This task involves:

    1. Conducting comprehensive research using authorized scholarship databases and official websites.
    2. Analyzing complex eligibility criteria and matching them to individual student profiles.
    3. Verifying the legitimacy and current status of each scholarship opportunity.
    4. Providing detailed, accurate, and up-to-date information on each scholarship.
    5. Offering insights on application processes and strategies to increase chances of success.
    6. Ensuring all information is presented in a clear, structured format for easy understanding.
    7. Maintaining the highest standards of ethical practice in scholarship recommendation and verification.

    My goal is to empower students with trustworthy, actionable scholarship information that aligns with their academic aspirations, financial needs, and eligibility criteria.

    For each query, I will identify and provide detailed information on at least 5 relevant scholarship opportunities.""",
    expected_output="""
    Provide detailed reports for at least 5 scholarship opportunities. Each scholarship should be presented in the following structured format:
        "name": "Full official name of the scholarship",
        "country": "Country offering the scholarship or 'International' if applicable",
        "amount": "Specific amount or range, including currency",
        "scholarship_form": "Direct link to the application form if available",
        "application_start_date": "YYYY-MM-DD format",
        "application_end_date": "YYYY-MM-DD format",
        "deadline": "YYYY-MM-DD format, with any time zone specifications",
        "eligibility": "Concise list of key eligibility criteria",
        "field_of_study": "Specific fields or 'All fields' if applicable",
        "degree_level": "Undergraduate/Masters/PhD/Postdoctoral/Multiple",
        "duration": "Length of the scholarship support",
        "website": "Official scholarship webpage URL",
        "funding_type": "Full/Partial, with specifics on costs covered",
        "application_process": "Step-by-step overview of the application procedure",
        "is_legitimate": true/false,
        "is_current": true/false,
        "organization_verified": true/false,
        "application_process_verified": true/false,
        "potential_red_flags": ["List any concerns or suspicious elements"],
        "verification_methods_used": ["List of specific methods used for verification"],
        "last_verified_date": "YYYY-MM-DD",
        "overall_trust_score": "1-10 scale, with 10 being highest trust",
        "verification_notes": "Detailed explanation of verification process and findings"

    Provide this structured information for each of the at least 5 relevant scholarship opportunities identified. Ensure all fields are filled accurately, using 'N/A' for any information that is not available or not applicable. Prioritize scholarships that best match the student's profile and have the highest trust scores.

    After presenting the detailed information for each scholarship, provide a summary that includes:
    1. Total number of scholarships found
    2. Brief overview of the types of scholarships identified
    3. Any notable trends or patterns observed in the scholarship opportunities
    4. General advice for applicants based on the scholarships found
    """,
    agent=ScholarshipFinder_agent,
    tools=[search_tool, web_rag_tool]
    )

    search_crew = Crew(
        agents=[ScholarshipFinder_agent],
        tasks=[ScholarshipFinder_task],
        )
    search_result = search_crew.kickoff(inputs={'topic': str(prompt)})
    return search_result

def details(prompt):
    search_results = output(prompt)
    results = extract_information(search_results)
    detaiils = json.loads(results)
    with open('detained.json', 'w') as f:
        json.dump(detaiils, f, indent=4)
    return detaiils