#!bash
set -e
echo "SASS builder for static sites is on service now!"
echo "Build CSS ..."
node-sass --output-style compressed --output ../docs/assets/stylesheets ./stylesheets/extra.scss
