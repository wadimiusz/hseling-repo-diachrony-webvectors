FROM hseling/hseling-api-base:python3.6-alpine3.7 as build

LABEL maintainer="Vlad Mikhailov <info@linghub.ru>"

RUN mkdir /dependencies
COPY ./hseling_api_diachrony_webvectors/requirements.txt /dependencies/requirements.txt
COPY ./hseling_api_diachrony_webvectors/setup.py /dependencies/setup.py

RUN pip install -r /dependencies/requirements.txt

FROM hseling/hseling-api-base:python3.6-alpine3.7 as production

COPY --from=build /usr/local/lib/python3.6/site-packages /usr/local/lib/python3.6/site-packages
COPY --from=build /usr/lib/python3.6/site-packages /usr/lib/python3.6/site-packages

COPY --from=build /dependencies /dependencies

COPY ./hseling_lib_diachrony_webvectors /dependencies/hseling_lib_diachrony_webvectors
RUN pip install /dependencies/hseling_lib_diachrony_webvectors

COPY ./hseling_api_diachrony_webvectors/hseling_api_diachrony_webvectors /app/hseling_api_diachrony_webvectors
COPY ./hseling_api_diachrony_webvectors/main.py /app/main.py

ENV HSELING_API_YOUR_APPLICATION_SETTINGS="/app/hseling_api_diachrony_webvectors/default_settings.py"

HEALTHCHECK CMD curl --fail http://localhost/healthz || exit 1
