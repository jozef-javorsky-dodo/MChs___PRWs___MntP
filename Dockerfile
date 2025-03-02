WORKDIR /src/bcknd/app

COPY . ./

ENV BUNDLE_FROZEN=true

RUN bundle config set --local without 'test'
RUN bundle install

CMD ["python", "./mchs___prws___mntp.py"]
