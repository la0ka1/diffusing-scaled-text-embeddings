# Local preview of the project page, without installing Ruby.
#   docker compose up            # build (first time) + serve at http://localhost:4000/diffusing-scaled-text-embeddings/
#   docker compose up --build    # rebuild the image after changing the Gemfile
#   docker compose down          # stop
# GitHub Pages builds with Ruby 3.3 and the github-pages gem, so the image does the same.
FROM ruby:3.3

WORKDIR /srv/site

# Native extensions (eventmachine, ffi, nokogiri, ...) need a compiler.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# The lock file may come from another CPU architecture; register both Linux
# platforms so it works on Apple Silicon and on Intel machines.
COPY Gemfile Gemfile.lock* ./
RUN bundle lock --add-platform aarch64-linux x86_64-linux \
 && bundle install

COPY . .

# 4000 = site, 35729 = livereload websocket
EXPOSE 4000 35729

# --force_polling: file-change detection on bind mounts (macOS/Windows)
# --livereload:    browser auto-refresh on rebuild
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--livereload", "--force_polling"]
