var express = require('express');
var app = express();
var fs = require("fs");
var rp = require('request-promise');
var createCsvWriter = require('csv-writer').createObjectCsvWriter;
const csvWriter = createCsvWriter({
    path: './data-set/dataset.csv',
    header: [
        {id: 'created_at', title: 'created_at'},
        {id: 'sensor_location', title: 'senor_location'},
        {id: 'streamHeight', title: 'streamHeight'},
        {id: 'updated_at', title: 'updated_at'}
    ]
});

app.get('/make_data',async function (req, res) {
    try {
        let records = [];
        const locations = [
          'Banteay Meanchey',
          'Battambang',
          'Phnom Penh',
          'Siemreap',
          'Pursat',
          'Kampong Thom',
          'Kampong Chhnang',
          'Kampot',
          'Phnom Penh',
          'Preah Sihanouk',
          'Preah Sihanouk',
          'Koh Kong',
          'Phnom Penh',
          'Koh Kong',
          'Phnom Penh',
          'Phnom Penh',
          'Phnom Penh',
          'Phnom Penh',
          'Phnom Penh',
          'Phnom Penh',
          'Phnom Penh',
          'Phnom Penh',
        ];
        for (let i = 1; i <= 50000; i += 1) {
          const url = `https://dashboard.ews1294.info/api/sensor_events?page=${i}`;
          const options = {
            uri: url,
            headers: {
              Accept: 'application/json',
              'Content-Type': 'application/json',
            },
            method: 'GET',
            auth: {
              user: '75b0ad212a2100aed17037af9fb4f0925c5364f8e979c99a044ee30a2a821160',
              pass: '',
            },
            json: true, // Automatically parses the JSON string in the response
          };
          let result = await rp(options);
          result = JSON.parse(JSON.stringify(result));
          res.end(JSON.stringify({status:'in processing...'}));
          if (result.length <= 0) { break; }
        for (const key in result) {
            const data = result[key];
            records.push({
            created_at: data.created_at || '',
            sensor_location: locations[data.sensor_id - 1],
            streamHeight: data.payload.streamHeight || '',
            updated_at: data.updated_at || '',
            });
        }
        console.log(`Total finished page : ${i}`);
        }
        csvWriter.writeRecords(records)
            .then(() => {
                console.log('--> Done');
            });
      } catch (e) {
        console.log('error : ', e);
        // response.exception(this.res, e);
      }
})

var server = app.listen(8081, function () {
   var host = server.address().address
   var port = server.address().port
   console.log("app listening at http://%s:%s", host, port)
})