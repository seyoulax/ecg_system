<h1 align=center><img src="https://aucentr.ru/wp-content/uploads/2020/10/%D0%9B%D0%BE%D0%B3%D0%BE-%D0%91%D0%92_1_png-1024x294.png" width="30%"/> <br/> Большие Вызовы. Проект CardioScreen <img  src="https://i.ibb.co/q5Qwv7g/logo.png" width=5% alt="logo" border="0"></h1>
<!-- <img src="https://aucentr.ru/wp-content/uploads/2020/10/%D0%9B%D0%BE%D0%B3%D0%BE-%D0%91%D0%92_1_png-1024x294.png" width="20%"/> -->

Данный проект реализован в рамках подготовки к конкурсу <b>` Большие Вызовы. Направление - Большие данные и машинное обучение ` </b>.

### *Ссылка на материалы:*

* [```работа```](https://drive.google.com/file/d/1fdEyHsHs7cVE0lBbjuFrIFzZnazVPCGC/view?usp=sharing) с описанием проекта
- [```презентация```](https://drive.google.com/file/d/1rVcetQ-hO00IkXJFkM_7nIQgsmjeYX4Y/view?usp=sharing) проекта

### *Структура репозитория:*

* [```preprocessing```](https://github.com/seyoulax/ecg_system/tree/325bb2a64ff98cb64d41b4daa7d08b1a42964d23/preprocessing "перейти в папку") - папка с кодом препроцессинга данных
* [```web```](https://github.com/seyoulax/ecg_system/tree/325bb2a64ff98cb64d41b4daa7d08b1a42964d23/web "перейти в папку") - папка с кодом веб-сайта
* [```inference_example.ipynb```](https://github.com/seyoulax/ecg_system/blob/325bb2a64ff98cb64d41b4daa7d08b1a42964d23/inference_example.ipynb "перейти в файл") - пример инференса модели
* [```training```](https://github.com/seyoulax/ecg_system/tree/325bb2a64ff98cb64d41b4daa7d08b1a42964d23/training "перейти в папку") - папка с кодом обучения моделей, представленных в работе

### *Инструкция по запускy веб-интерфейса:*

* перейти в папку web
* Загрузить и распаковать zip архив  с помощью команды `wget http://site.m1r0.webtm.ru:8080/s/ex6dKfgaEqLRJpg/download/models.zip`
* установть poetry `curl -sSL https://install.python-poetry.org | python3 -`
* `poetry install`
* `unzip models.zip`
* `poetry run python app.py`

### *!UPD:*

* в рамках разроботки данного проекта, был реализован ***[python module](https://github.com/seyoulax/ecg_worker)*** для работы с ЭКГ: ***обучением различных моделей***, ***препроцессингом данных***
