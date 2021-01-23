# A simple and stupid dashboard for e-mission

Issues: Since this repository is part of a larger project, all issues are tracked in the central docs repository. If you have a question, as suggested by the open source guide, please file an issue instead of sending an email. Since issues are public, other contributors can try to answer the question and benefit from the answer.

Dashboards! They are fairly essential for user acceptance, but there are many options to build them.
And the choice of the technology stack for them is particularly fraught.
And for community projects, especially outside computer stack, choosing a technology stack ensures that half your collaborators cannot access it.
For example, choosing python will cause R users to balk and vice versa.
And there is although contributors can write some javascript, picking a charting library again steepens the learning curve.

So we are going to use a simple and stupid dashboard.
This will consist of a reactive grid layout
(e.g. https://strml.github.io/react-grid-layout/examples/15-drag-from-outside.html)
served by a simple static express server following the instructions at
https://www.thoughts-in-motion.com/articles/creating-a-static-web-server-with-node-js-and-express/

The grid layout will display static, pre-generated images using whatever program the user wishes.
The program should take the time range as input and generate a static image shared with the express server.
We have included python examples using ipython notebook and simple python scripts for the following metrics:

- mode share (notebook)
- purpose share (notebook)
- total number of trips per day (python)
