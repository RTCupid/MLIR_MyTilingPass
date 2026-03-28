<div align="center">

# Реализация прохода тайлинга для операций матричного умножения в диалекте Linalg
  ![C++](https://img.shields.io/badge/C++-20-blue?style=for-the-badge&logo=cplusplus)
  ![CMake](https://img.shields.io/badge/CMake-3.20+-green?style=for-the-badge&logo=cmake)
  ![MLIR](https://img.shields.io/badge/MLIR-18.1+-yellow?style=for-the-badge&logo=llvm&logoColor=white)
  ![LLVM](https://img.shields.io/badge/LLVM-18.1+-blue?style=for-the-badge&logo=llvm)

</div>

## README на других языках

1. [Русский](/README-R.md)
2. [English](/README.md)

## Оглавление
- [0. Запуск программы](#запуск-программы)
- [1. Аннотация](#аннотация)
- [2. Введение](#введение)
- [3. Методика](#методика)
- [4. Реализация прохода тайлинга](#реализация-прохода-тайлинга)
- [5. Результаты](#результаты)
- [6. Заключение](#заключение)
- [7. Структура проекта](#структура-проекта)
- [Автор проекта](#автор-проекта)

## Запуск программы

Проект предполагает существование собранного проекта `MLIR/LLVM` с наличием  доступа к `MLIRConfig.cmake`. Клонирование и сборка осуществляются следующими командами:

```bash
git clone git@github.com:RTCupid/MLIR_MyTilingPass.git
cd MLIR_MyTilingPass
cmake -S . -B build -DMLIR_DIR=/path/to/lib/cmake/mlir
cmake --build build
```

Запуск программы производится в приведённом ниже формате:

```bash
cd build/bin
./my-tiling-opt <mlir_program>
```

Разработанный инструмент включает возможность задавать размерности тайлинга `M`, `N`, `K` при помощи следующей опции `-my-tiling="tile-sizes=%s,%s,%s`".

## Аннотация
Разработан инструмент на основе `mlir-opt`, который выполняет разбиение на блоки (`tiling`) для операций умножения матриц в диалекте `Linalg`. В реализации использована спецификация `TableGen` и класс на `C++`, производный от `MyTilingPassBase`, сгенерированного TableGen. Для определения подходящих шаблонов использован `TilingInterface`, а для выполнения преобразования с разбиением на блоки - `scf::tileUsingSCFForOp`.

Инструмент успешно протестирован на широком спектре конфигураций: матрицах с размерностями, кратными и не кратными размерам тайлов; динамических формах; двух последовательных операциях умножения матриц; прямоугольных матрицах с различными значениями `M`, `N`, `K`; `тензорах` и `memref`; а также для операций `linalg.matmul` и `linalg.generic`. Дополнительно проверена корректность работы в случае, когда операция умножения матриц изначально находится внутри цикла, — этот сценарий не поддерживался в первой версии.


## Введение
В тензорных компиляторах для глубокого обучения умножение матриц (`matmul`) доминирует по вычислительной нагрузке, определяя производительность обучения и инференса. Без учёта иерархии памяти генерация кода приводит к избыточным обращениям к `DRAM`, неэффективному использованию кэша и простою вычислительных ядер.

Наивная реализация с вложенными циклами не переупорядочивает доступ к данным. Каждое обращение загружает целую кэш-линию, а многократные проходы по одним и тем же данным порождают огромное число лишних пересылок между уровнями памяти. Повторное использование данных практически отсутствует, производительность упирается в пропускную способность памяти, а векторные инструкции и параллелизм на ядрах остаются незадействованными.

Предлагаемый проход тайлинга решает эти проблемы за счёт блочного разбиения операции. Размеры тайлов выбираются с учётом ёмкости кэша, что максимизирует повторное использование данных и сокращает промахи кэша. Кроме того, пасс автоматизирует трансформацию: разработчику не требуется вручную писать сложные вложенные циклы, а универсальность подхода позволяет корректно обрабатывать динамические формы, прямоугольные матрицы и случаи, когда `matmul` уже находится внутри цикла.

## Методика
Для реализации прохода тайлинга выбрана инфраструктура `MLIR`, предоставляющая гибкое внутреннее представление (`IR`) и развитую систему преобразований. Поскольку операции умножения матриц входят в состав диалекта `Linalg`, разрабатываемый оптимизирующий проход (`pass`) ориентирован именно на этот диалект.

В `MLIR` существуют два основных подхода к реализации подобных преобразований.

Первым является паттерн‑ориентированный подход, подразумевающий создание набора классов, наследующих `OpRewritePattern`, и применение их через механизмы перезаписи, такие как `applyPatternsAndFoldGreedily`. Этот метод удобен при необходимости комбинировать несколько связанных преобразований, но требует явного управления очередностью применения паттернов.

Вторым подходом к задаче является итеративный обход операций, а именно ручная итерация по всем операциям модуля с проверкой реализации интерфейса `TilingInterface`. Данный подход обеспечивает полный контроль над процессом преобразования и упрощает отладку, так как не полагается на недетерминированный обход паттернов.

В рамках данной работы выбран второй подход, позволяющий точно отслеживать применение тайлинга к каждой операции `linalg.matmul` и `linalg.generic`, реализующей умножение матриц. Для выполнения тайлинга используется доступная в `MLIR` функция `scf::tileUsingSCFForOp`, которая принимает операцию, поддерживающую `TilingInterface`, и возвращает новые циклы `scf.for` с вложенными операциями над блоками.

Сборка проекта организована в `standalone` конфигурации, что позволяет изолировать разработку тестового пасса от основного дерева `MLIR` и упрощает его интеграцию в сторонние инструменты.

## Реализация прохода тайлинга
Для написания первой итерации прохода изучена структура диалектов и проходов в `MLIR` (см. [Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)), и реализована минимальная версия своего диалекта и преобразования для его операции (см. [MLIR_MyDialect](https://github.com/RTCupid/MLIR_MyDialect)). Основные проблемы при этом вызвала установка зависимостей между файлами в `CMakeLists.txt` и определение доступного набора уже реализованных функций.

Первая итерация прохода тайлинга реализована с использованием паттернов переписывания (`RewritePattern`). Такой подход позволил задействовать готовые механизмы, например, `applyPatternsAndFoldGreedily`, для многократного применения преобразований. 

Полученный проход успешно проходил большую часть тестов. Однако возникла проблема запуска прохода на уже обработанных операциях. проблема была решена ограничением количества итераций в `GreedyRewriteConfig` и добавлением эвристической проверки, что операция не находится в цикле. На этом этапе возникли трудности с умножением матриц, которое изначально находилось внутри цикла. Принято решение использовать подход на основе обхода операций (`walk`) и прямого вызова `scf::tileUsingSCFForOp`. Это позволило точнее контролировать работу с операциями и проще реализовать одиночный проход по операциям, удовлетворяющим `TilingInterface`.

Для генерации базового класса разработано его описание на языке `TableGen` (см. [Passes.td](/include/MyTiling/Passes.td)).

<details>
<summary>Описание для TableGen:</summary>

```mlir
def MyTilingPass : Pass<"my-tiling", "mlir::func::FuncOp"> {
  let summary = "Tile linalg operations with user-specified tile sizes";
  let description = [{
    Applies tiling transformation to linalg operations using TilingInterface.
    Takes tile sizes as pass option.
  }];

  let constructor = "mlir::createMyTilingPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::scf::SCFDialect",
                           "mlir::memref::MemRefDialect", "mlir::tensor::TensorDialect"];

  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
          "Tile sizes for each loop dimension">
  ];
}
```

</details>

По этому описанию генерировался класс `MyTilingPassBase`, содержащий такие базовые методы, как `getArgument`, `getDescription`, `getDependentDialects` и другие, а также функция для регистрации прохода `registerMyTilingPass`.

На `C++` реализован класс `MyTilingPass` (см. [MyTiling.cpp](/lib/Transforms/MyTilingPass.cpp)), наследующийся от базового, который реализует основную логику прохода - `runOnOperation`.

<details>
<summary>Определение runOnOperation:</summary>

```c++
void MyTilingPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = &getContext();
  Builder b(ctx);

  SmallVector<OpFoldResult> tileSizesOFR =
      llvm::to_vector(llvm::map_range(this->tileSizes, [&](int64_t v) -> OpFoldResult {
        return b.getIndexAttr(v);
      }));
  if (tileSizesOFR.empty()) {
    tileSizesOFR = {b.getIndexAttr(32), b.getIndexAttr(32), b.getIndexAttr(32)};
  }

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizesOFR);

  SmallVector<TilingInterface> worklist;
  func.walk([&](TilingInterface op) {
    worklist.push_back(op);
  });

  IRRewriter rewriter(ctx);
  for (TilingInterface op : worklist) {
    if (op->getBlock() == nullptr) continue;

    FailureOr<scf::SCFTilingResult> result =
      scf::tileUsingSCFForOp(rewriter, op, tilingOptions);
    if (succeeded(result)) {
      rewriter.replaceOp(op, result->replacements);
    }
  }
}

```

Функция получает все операции, реализующие TilingInterface, и для каждой применяет scf::tileUsingSCFForOp с заданными размерами тайлов

</details>

Для создания инструмента тайлинга разработана функция `main` с регистрацией диалектов и прохода тайлинга (см. [my-tiling-opt.cpp](/tools/my-tiling-opt/my-tiling-opt.cpp)).

<details>
<summary>Функция main:</summary>

```c++
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::my_tiling::registerMyTilingPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv,
                        "My Tiling Optimization Tool",
                        registry));
}

```

</details>


## Результаты

## Заключение

## Структура проекта

## Автор проекта

<div align="center">

  <a href="https://github.com/RTCupid">
    <img src="https://raw.githubusercontent.com/BulgakovDmitry/3D_triangles/main/img/A.jpeg" width="160" height="160" style="border-radius: 50%;">
  </a>
  <br>
  <a href="https://github.com/RTCupid"><strong>@RTCupid </strong></a>
  <br>
</div>
