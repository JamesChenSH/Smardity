# Smardity
This is a project designed for CSC2125H 2025S

myapp/                  (Contains the entire Express app)
├─ app.js               The core logic of the Express app.
├─ bin/                 (Contains the app's executable scripts)
│  └─ www               A wrapper that runs app.js.
├─ node_modules/        (Contains dependencies installed by npm)
├─ package-lock.json    JSON manifest of installed dependencies.
├─ package.json         JSON of dependencies and config specific to your app.
├─ public/              (Files downloaded by the user's web browser)
│  ├─ images/           (Contains client-accessible image files)
│  ├─ javascripts/      (Contains client-accessible JavaScript files)
│  └─ stylesheets/      (Contains client-accessible CSS)
│     └─ style.css      The site's CSS stylesheet.
├─ routes/              (Contains logic for individual site routes)
│  ├─ index.js          Logic of the "index" route (/).
│  └─ users.js          Logic of the "users" route (/users).
└─ views/               (Contains HTML templates)
   ├─ error.pug         View displayed for error pages, such as HTML 404.
   ├─ index.pug         View displayed for the site root (/).
   └─ layout.pug        View template of layout shared by all pages.