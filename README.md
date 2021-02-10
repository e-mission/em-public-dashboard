# A simple and stupid dashboard for e-mission

Issues: Since this repository is part of a larger project, all issues are tracked in the central docs repository. If you have a question, as suggested by the open source guide, please file an issue instead of sending an email. Since issues are public, other contributors can try to answer the question and benefit from the answer.

## Development

We use docker images for the software dependencies since we will not be modifying them here.

So the steps are:

#### Launch dev environment

```
$ docker-compose -f docker-compose.dev.yml  up
Creating network "em-public-dashboard_emission" with the default driver
Creating em-public-dashboard_db_1 ... done
Creating em-public-dashboard_plot-gen_1  ... done
Creating em-public-dashboard_dashboard_1 ... done
...
dashboard_1  | Starting up http-server, serving ./
dashboard_1  | Available on:
dashboard_1  |   http://127.0.0.1:8080
dashboard_1  |   http://172.25.0.3:8080
dashboard_1  | Hit CTRL-C to stop the server
...
notebook-server_1  |
notebook-server_1  |     To access the notebook, open this file in a browser:
notebook-server_1  |         file:///root/.local/share/jupyter/runtime/nbserver-22-open.html
notebook-server_1  |     Or copy and paste one of these URLs:
notebook-server_1  |         http://f8317197efaf:8888/?token=5cfd541b7461a47310c9c8aaa4114f921457a6f17b8ca159
notebook-server_1  |      or http://127.0.0.1:8888/?token=5cfd541b7461a47310c9c8aaa4114f921457a6f17b8ca159
...
```

#### Test the frontend install

Go to http://localhost:3274/ to see the front-end. Note that the port is *3274*
instead of the *8080* in the logs, since we remap it as part of the docker-compose.

#### Test the notebook install

Use the notebook URL from the console but change `8888` to `47962`

```
http://127.0.0.1:8888/?token=<token>
```

becomes

```
http://127.0.0.1:47962/?token=<token>
```

#### Load some data

https://github.com/e-mission/e-mission-server/#quick-start

There are multiple sources listed there, or you can use the mongodump from:
https://github.com/asiripanich/emdash#loading-test-data

#### Loading data from a mongodump

We have a helper script to load data directly from a mongodump.

```
$ bash viz_scripts/docker/load_mongodump.sh <mongodump_file>
```

Note that this expects a standard setup with:
- this repository checked out under the `em-public-dashboard` directory, which makes the database name `em-public-dashboard_db_1`
- the incoming mongodump is in tar gz format. This should be true of all canbikeco dumps, you may need to change the `tar xvf` to `unzip` otherwise.  The mongo container typically doesn't have zip installed, so using tar is more portable.

**If you have a non-standard setup, please use your expertise to change the script appropriately.**

#### Happy visualizations!

Look at the existing notebooks for examples on how to start.
In particular, before you check in, please make sure that you are reading
inputs correctly, because otherwise, no metrics will be generated.

### Design decisions

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

In order to get the prototype out, there are a lot of shortcuts. We can revisit
this later if there is sufficient interest/funding.

- Using gridster (https://github.com/dsmorse/gridster.js/) and bootstrap instead of react
- Using the pre-built (https://hub.docker.com/r/danjellz/http-server) instead of express
- Using a mounted volume instead of building a custom docker image to make deployment easier
- Using the e-mission server codebase to generate graphs instead of a REST API

The one part where we are NOT cutting corners is in the parts where we expect
contributions from others. We are going to build in automated tests for that
part to ensure non-bitrotted code.
