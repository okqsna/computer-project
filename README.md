# Бібліотека для роботи з графами 
### Комп'ютерний проєкт з дискретної математики 

<i>Підготували Юрій Фітьо, Катерина Сучок, Дарина Ничипорук, Юліан-Володимир Заєць, Оксана Москв'як</i>

Дана бібліотека опрацьовує отримані на вхід графи,
застосовує до них Гамільтоновий цикл, цикл Ейлера, <br>
перевіряє на двочастковість, перевіряє на ізоморфність 
та розфарбовує зв'язний граф у 3 кольори.

Детальніше кожну із функцій розглянуто нижче:


### Гамільтоновий цикл






### Цикл Ейлера



<hr>

### Двочастковість графу

#### Функція ``` bipartite_graph_check ``` перевіряє чи граф є двочастковим.

Основною ідеєю реалізації даної функції є використання <b>алгоритму пошуку в ширину (BFS)</b> та <b>розфарбовування вершин графа у два кольори</b> (тобто реалізація розділення вершин графа на 2 множини, у яких не повторюються вершини). 

Допоміжна функція ``` to_oriented ``` перетворює граф з неорієнтованого в орієнтований для подальшої роботи.

Допоміжна функція ``` get_neighbouring_values ``` знаходить для кожної вершини сусідські та повертає словник, де ключ це <b>вершина</b>, а значення - <b>множина сусідських вершин</b>.

На початку функція створює <b>“чергу”</b> ``` not_visited_vertices ``` для зафарбовування вершин із цієї черги та словник, який зберігає колір вершин. Імплементовано цикл ``` while ```, який працює поки черга не буде пустою. Для подальшої перевірки дістаємо першу вершину з черги, для якої відбувається перевірка чи не є ця вершина в відвіданих, далі відбувається перевірка сусідніх до початкової вершин. Якщо поточна вершина та сусідська вершини однакового кольору - то цей граф <b>не двочастковий</b>, функція повертає значення - ``` False ```

Якщо всі вершини було зафарбовано без збігів, то граф <b>двочастковий</b> . Функція повертає значення - ``` True ```

<hr>

### Ізоморфність графу


###  Розфарбування зв'язного графу в 3 кольори



