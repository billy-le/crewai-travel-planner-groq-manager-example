import os
from crewai import Crew, Process
from trip_agents import TripAgents
from trip_tasks import TripTasks
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()


class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.cities = cities
        self.origin = origin
        self.interests = interests
        self.date_range = date_range

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()

        identify_task = tasks.identify_task(
            city_selector_agent,
            self.origin,
            self.cities,
            self.interests,
            self.date_range,
        )
        gather_task = tasks.gather_task(
            local_expert_agent, self.origin, self.interests, self.date_range
        )
        plan_task = tasks.plan_task(
            travel_concierge_agent,
            self.origin,
            self.interests,
            self.date_range,
            [identify_task, gather_task],
        )

        crew = Crew(
            agents=[city_selector_agent, local_expert_agent, travel_concierge_agent],
            tasks=[plan_task],
            verbose=True,
            process=Process.hierarchical,
            manager_llm=ChatGroq(
                temperature=0,
                model_name="llama3-8b-8192",
                groq_api_key=os.getenv("GROQ_API_KEY"),
            ),
            max_rpm=25,
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    print("## Welcome to Trip Planner Crew")
    print("-------------------------------")
    location = "vietnam"
    cities = "manila, phuket"
    date_range = "may 28, 2024 - may 31, 2024"
    interests = "beach, sight-seeing, eating local food"
    trip_crew = TripCrew(location, cities, date_range, interests)
    result = trip_crew.run()
    print("\n\n########################")
    print("## Here is you Trip Plan")
    print("########################\n")
    print(result)
