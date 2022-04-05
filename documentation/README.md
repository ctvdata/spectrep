### Crear documentación

```
cd documentation
sphinx-apidoc -o ./_modules ../spectraltrep/
make html
```

### Ver documentación
```
cd documentation
firefox _build/html/index.html
```