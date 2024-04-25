<h1 align=center><img src="https://aucentr.ru/wp-content/uploads/2020/10/%D0%9B%D0%BE%D0%B3%D0%BE-%D0%91%D0%92_1_png-1024x294.png" width="30%"/> <br/> Большие Вызовы. Проект CardioScreen <img  src="https://i.ibb.co/q5Qwv7g/logo.png" width=5% alt="logo" border="0"></h1>
<!-- <img src="https://aucentr.ru/wp-content/uploads/2020/10/%D0%9B%D0%BE%D0%B3%D0%BE-%D0%91%D0%92_1_png-1024x294.png" width="20%"/> -->

Данный проект реализован в рамках подготовки к конкурсу <b>` Большие Вызовы. Направление - Большие данные и машинное обучение ` </b>

<h3>Структура репозитория</h3>

* [```process```](https://github.com/XXXM1R0XXX/final2/tree/b2c27b43e21dbc5ec4dfeceb6c71d734852672b9/process "перейти в папку") - папка с кодом препроцессинга данных
* [```web```](https://github.com/XXXM1R0XXX/final2/tree/dcd84f38b0393c7fe9dec1a7e1064d3fa53e2e5a/web "перейти в папку") - папка с кодом веб-сайта
* [```inference_example.ipynb```](https://github.com/XXXM1R0XXX/final2/blob/dcd84f38b0393c7fe9dec1a7e1064d3fa53e2e5a/inference_example.ipynb "перейти в файл") - пример инференса модели
* [```training_example.ipynb```](https://github.com/XXXM1R0XXX/final2/blob/dcd84f38b0393c7fe9dec1a7e1064d3fa53e2e5a/training_example.ipynb "перейти в файл") - пример обучения модели

<h3>Инструкция по запускy веб-интерфейса</h3>

* перейти в папку web
* Загрузить и распаковать zip архив  с помощью команды `wget http://site.m1r0.webtm.ru:8080/s/ex6dKfgaEqLRJpg/download/models.zip`
* установть poetry `curl -sSL https://install.python-poetry.org | python3 -`
* `poetry install`
* `unzip models.zip`
* `poetry run python app.py`