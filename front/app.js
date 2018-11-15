const express = require('express');
const path = require('path');
const proxy = require('express-http-proxy');
const app = express();
const port = 3000;

app.use('/', express.static(path.join(__dirname, 'static')))

app.use('/', proxy(process.env.ENV_API || 'http://localhost:8000', {
    limit: '120mb'
}));

app.listen(port, () => console.log(`App listening on port ${port}!`));