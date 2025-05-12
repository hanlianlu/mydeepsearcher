import unittest
from deepsearcher.agent import prompt
from deepsearcher.tools import log

class TestPrompt(unittest.TestCase):
    def setUp(self):
        log.set_dev_mode(True)
    
    def test_search_prompt(self):
        question = "What is the meaning of life?"
        collection_names = [
            "Philosophical Literature",
            "Psychological Research",
            "Cultural and Religious Beliefs",
            ]
        collection_descriptions = [
            "Contains discussions and theories on the meaning of life by historical philosophers, including excerpts from ancient and modern philosophical works",
            "Covers studies on human well-being, sense of purpose, and life satisfaction, analyzing how these factors influence people's understanding of life's meaning.",
            "Collects perspectives on the meaning of life from various cultures and religions, showcasing the differences in their belief systems and values.",
            ]
        context = None
        collection_info = [{"collection_name": collection_names[i], "collection_description": collection_descriptions[i]} for i in range(len(collection_names))]
        p = prompt.get_vector_db_search_prompt(question, collection_info, context)
        log.info("\n" + p)

        """ prompt llm output example:
{
    "Philosophical Literature": "What insights do historical philosophers provide about the purpose and meaning of life?",
    "Psychological Research": "How do studies on human well-being and sense of purpose contribute to our understanding of life's meaning?",
    "Cultural and Religious Beliefs": "What are the varying beliefs about the meaning of life across different cultures and religions?"
}
        """
    
    def test_reflect_prompt(self):
        question = "What is the meaning of life?"
        collections = [
            "Philosophical Literature",
            "Psychological Research",
            "Cultural and Religious Beliefs",
            ]
        mini_questions = [
            "What insights do historical philosophers provide about the purpose and meaning of life?",
            "How do studies on human well-being and sense of purpose contribute to our understanding of life's meaning?",
            "What are the varying beliefs about the meaning of life across different cultures and religions?",
            ]
        mini_chuncks = [
            "Philosophers like Aristotle emphasized eudaimonia (flourishing) through virtue, while existentialists like Sartre argued that meaning is self-created through choices. Nietzsche proposed that life's meaning lies in pursuing significant goals, and Stoics like Zeno focused on virtue and self-control.",
            "Studies suggest that a sense of purpose enhances well-being, linking meaningfulness to active engagement with worthwhile activities. Viktor Frankl's logotherapy highlights finding meaning through personal values and relationships, even in adversity.",
            "Eastern philosophies like Buddhism seek enlightenment to escape suffering, while Abrahamic religions emphasize serving God. Taoism advocates harmony with nature, and existentialism posits that individuals must create their own meaning.",
            ]
        p = prompt.get_reflect_prompt(question, collections, mini_questions, mini_chuncks)
        log.info("\n" + p)

        """ prompt llm output example: gpt-4o-mini
{
    "related_content": {
        "What insights do historical philosophers provide about the purpose and meaning of life?": "YES",
        "How do studies on human well-being and sense of purpose contribute to our understanding of life's meaning?": "YES",
        "What are the varying beliefs about the meaning of life across different cultures and religions?": "YES"
    },
    "miss_questions": [
        "What role do personal experiences play in shaping an individual's understanding of life's meaning?",
        "How do contemporary thinkers and scientists interpret the meaning of life?",
        "What impact do societal and environmental factors have on people's perceptions of life's meaning?"
    ]
}
        """

        """prompt llm output example: gpt-4o
{}
        """   
    
    def test_final_answer_prompt(self):
        question = "What is the meaning of life?"
        collections = [
            "Philosophical Literature",
            "Psychological Research",
            "Cultural and Religious Beliefs",
            ]
        mini_questions = [
            "What insights do historical philosophers provide about the purpose and meaning of life?",
            "How do studies on human well-being and sense of purpose contribute to our understanding of life's meaning?",
            "What are the varying beliefs about the meaning of life across different cultures and religions?",
            ]
        mini_chuncks = [
            "Philosophers like Aristotle emphasized eudaimonia (flourishing) through virtue, while existentialists like Sartre argued that meaning is self-created through choices. Nietzsche proposed that life's meaning lies in pursuing significant goals, and Stoics like Zeno focused on virtue and self-control.",
            "Studies suggest that a sense of purpose enhances well-being, linking meaningfulness to active engagement with worthwhile activities. Viktor Frankl's logotherapy highlights finding meaning through personal values and relationships, even in adversity.",
            "Eastern philosophies like Buddhism seek enlightenment to escape suffering, while Abrahamic religions emphasize serving God. Taoism advocates harmony with nature, and existentialism posits that individuals must create their own meaning.",
            ]
        p = prompt.get_final_answer_prompt(question, collections, mini_questions, mini_chuncks)
        log.info("\n" + p)

        """ prompt llm output example: gpt-4o-mini
The meaning of life varies across philosophical, psychological, and cultural perspectives. 

1. **Philosophers** like Aristotle focus on eudaimonia, or flourishing through virtue, while existentialists such as Sartre argue that meaning is self-created through personal choices. Nietzsche believes life's meaning lies in pursuing significant goals, and Stoics emphasize virtue and self-control. (Source: Philosophical Literature)

2. **Psychological research** indicates that a sense of purpose enhances well-being and is linked to engagement in meaningful activities. Viktor Frankl's logotherapy stresses finding meaning through personal values and relationships, even amidst adversity. (Source: Psychological Research)

3. **Cultural and religious beliefs** show diverse views: Eastern philosophies like Buddhism aim for enlightenment to escape suffering, Abrahamic religions focus on serving God, Taoism seeks harmony with nature, and existentialism posits that individuals must create their own meaning. (Source: Cultural and Religious Beliefs)
        """
        
        """prompt llm output example: gpt-4o
The meaning of life has been explored through various philosophical, psychological, and cultural perspectives. According to Philosophical Literature, Aristotle emphasized eudaimonia (flourishing) through virtue, Sartre argued that meaning is self-created through choices, Nietzsche saw meaning in pursuing significant goals, and Stoics like Zeno focused on virtue and self-control. Psychological Research suggests that a sense of purpose enhances well-being, with Viktor Frankl's logotherapy emphasizing meaning through personal values and relationships, even in adversity. Cultural and Religious Beliefs highlight different views, such as Buddhism seeking enlightenment to escape suffering, Abrahamic religions focusing on serving God, Taoism advocating harmony with nature, and existentialism asserting that individuals must create their own meaning. 
        """