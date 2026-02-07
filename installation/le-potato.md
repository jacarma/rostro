# Le Potato (AML-S905X-CC) — Guia de instalacion completa

## Hardware necesario

- Libre Computer Le Potato (AML-S905X-CC) — 2GB RAM
- Micro SD (16GB minimo, clase 10 o superior)
- Cable HDMI (para verificacion visual)
- Cable Ethernet (conexion a red local)
- Fuente de alimentacion 5V/2A micro USB
- Desde tu ordenador: lector de micro SD (o adaptador)
- Microfono USB (imprescindible para usar Rostro, opcional solo durante la instalacion)

## Paso 1: Flashear Armbian en la micro SD

1. Descargar **Armbian** para Le Potato desde:
   https://www.armbian.com/lepotato/

   Elegir la imagen **minimal/CLI** (sin escritorio). Ejemplo:
   `Armbian_community_25.2.0-trunk.436_Lepotato_trixie_current_6.12.58_minimal.img.xz`

2. Flashear con **balenaEtcher** (o `dd`):
   - Insertar la micro SD en tu ordenador
   - Abrir balenaEtcher: https://etcher.balena.io/
   - Seleccionar la imagen `.img.xz` descargada
   - Seleccionar la micro SD
   - Flash

3. Insertar la micro SD en la Le Potato, conectar Ethernet y alimentacion.

## Paso 2: Primer arranque de Armbian

Armbian tiene un asistente interactivo en el primer arranque. Se accede por SSH:

```bash
# Buscar la IP de la Le Potato (revisar router o usar nmap)
nmap -sn 192.168.0.0/24 | grep -B2 "libre"

# Conectar por SSH (usuario: root, password: 1234)
ssh root@<IP_LE_POTATO>
```

El asistente pedira:
- **Nueva contrasena de root**
- **Crear usuario normal** (ej: nombre `rostro`)
- **Shell**: bash
- **Locale**: puede dejarse por defecto

> **Nota:** Guardar las credenciales en `.env` del proyecto:
> ```
> ROOT_PASSWORD=<password>
> USER_ACCOUNT=rostro
> USER_ACCOUNT_PASSWORD=<password>
> ```

## Paso 3: Configurar SSH con claves (desde tu ordenador)

Esto permite conectar sin escribir la contrasena cada vez.

```bash
# Limpiar clave vieja si la IP fue usada antes
ssh-keygen -R <IP_LE_POTATO>

# Copiar clave publica al usuario rostro
ssh-copy-id rostro@<IP_LE_POTATO>

# Copiar clave publica a root (necesario para la instalacion)
ssh-copy-id root@<IP_LE_POTATO>

# Verificar que funciona sin contrasena
ssh rostro@<IP_LE_POTATO> "echo OK"
```

## Paso 4: Seguridad basica

### 4.1 Configurar sudo sin contrasena (temporalmente, para la instalacion)

```bash
ssh root@<IP_LE_POTATO> 'echo "rostro ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/rostro && chmod 440 /etc/sudoers.d/rostro'
```

### 4.2 Asegurar SSH

```bash
ssh root@<IP_LE_POTATO> '
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
sed -i "s/^#*PermitRootLogin.*/PermitRootLogin no/" /etc/ssh/sshd_config
sed -i "s/^#*PasswordAuthentication.*/PasswordAuthentication no/" /etc/ssh/sshd_config
sed -i "s/^#*MaxAuthTries.*/MaxAuthTries 3/" /etc/ssh/sshd_config
sed -i "s/^#*LoginGraceTime.*/LoginGraceTime 30/" /etc/ssh/sshd_config
systemctl restart sshd
'
```

> **IMPORTANTE:** A partir de aqui, root por SSH queda deshabilitado.
> Todos los comandos siguientes se ejecutan como `rostro` con `sudo`.

Verificar que SSH sigue funcionando:
```bash
ssh rostro@<IP_LE_POTATO> "echo OK"
```

### 4.3 Firewall (UFW)

```bash
ssh rostro@<IP_LE_POTATO> '
sudo apt install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
echo "y" | sudo ufw enable
sudo ufw status
'
```

### 4.4 Fail2ban

```bash
ssh rostro@<IP_LE_POTATO> '
sudo apt install -y fail2ban
sudo bash -c "cat > /etc/fail2ban/jail.local << EOF
[sshd]
enabled = true
maxretry = 3
bantime = 3600
findtime = 600
EOF"
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
'
```

## Paso 5: Actualizar sistema

```bash
ssh rostro@<IP_LE_POTATO> 'sudo apt update && sudo apt upgrade -y'
```

> **Nota:** Esto puede tardar varios minutos. Si incluye un kernel update,
> la placa puede reiniciarse automaticamente.

## Paso 6: Actualizaciones automaticas de seguridad

```bash
ssh rostro@<IP_LE_POTATO> '
sudo apt install -y unattended-upgrades

sudo bash -c "cat > /etc/apt/apt.conf.d/20auto-upgrades << EOF
APT::Periodic::Update-Package-Lists \"1\";
APT::Periodic::Unattended-Upgrade \"1\";
EOF"

# Reboot automatico a las 4am si es necesario (kernel updates)
sudo sed -i "s|^//Unattended-Upgrade::Automatic-Reboot .*|Unattended-Upgrade::Automatic-Reboot \"true\";|" /etc/apt/apt.conf.d/50unattended-upgrades
sudo sed -i "s|^//Unattended-Upgrade::Automatic-Reboot-Time .*|Unattended-Upgrade::Automatic-Reboot-Time \"04:00\";|" /etc/apt/apt.conf.d/50unattended-upgrades
'
```

## Paso 7: Instalar paquetes base

```bash
ssh rostro@<IP_LE_POTATO> '
# Entorno grafico minimo
sudo apt install -y xorg openbox lightdm

# Audio
sudo apt install -y pulseaudio alsa-utils pavucontrol

# Python y multimedia
sudo apt install -y python3 python3-pip python3-venv ffmpeg

# Utilidades
sudo apt install -y curl wget htop git

# Para ocultar cursor y audio de sounddevice
sudo apt install -y unclutter libportaudio2
'
```

> **Nota:** Esta instalacion tarda 5-15 minutos en la Le Potato.

## Paso 8: Transferir el proyecto

Desde tu ordenador, en el directorio del proyecto:

```bash
rsync -avz \
  --exclude='venv' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.pytest_cache' \
  /ruta/a/rostro/ rostro@<IP_LE_POTATO>:/home/rostro/avatar/
```

Esto copia todo el codigo fuente incluyendo el `.env` con la API key.

## Paso 9: Instalar dependencias Python

```bash
ssh rostro@<IP_LE_POTATO> '
cd ~/avatar
python3 -m venv venv
source venv/bin/activate
pip install -e .
'
```

Esto instala todas las dependencias definidas en `pyproject.toml`:
openai, pygame, sounddevice, numpy, pyyaml, python-dotenv.

Verificar:
```bash
ssh rostro@<IP_LE_POTATO> '
~/avatar/venv/bin/python -c "
from rostro.runtime.controller import RuntimeController
from rostro.providers.llm.openai import OpenAILLMProvider
print(\"Imports OK\")
"
'
```

## Paso 10: Configurar modo kiosco

### 10.1 LightDM autologin

```bash
ssh rostro@<IP_LE_POTATO> '
sudo bash -c "cat > /etc/lightdm/lightdm.conf << EOF
[Seat:*]
autologin-user=rostro
autologin-session=openbox
user-session=openbox
EOF"

sudo groupadd -f autologin
sudo usermod -aG autologin rostro
sudo systemctl enable lightdm
'
```

### 10.2 Openbox autostart

```bash
ssh rostro@<IP_LE_POTATO> '
mkdir -p ~/.config/openbox
cat > ~/.config/openbox/autostart << '\''EOF'\''
# Activar HDMI como salida principal
xrandr --output HDMI-1-1 --mode 1920x1080 --primary --output None-1 --off

# Desactivar screen saver y DPMS
xset s off
xset -dpms
xset s noblank

# Ocultar cursor despues de 3 segundos
unclutter -idle 3 &

# Lanzar la app del avatar
bash -c "cd ~/avatar && . venv/bin/activate && python -m rostro.main" &
EOF
'
```

> **IMPORTANTE:** El autostart de Openbox usa `/bin/sh`, no bash.
> Por eso se usa `bash -c "..."` para la app y `. venv/bin/activate`
> en vez de `source` (que no existe en sh).
>
> El comando `xrandr` es necesario porque Xorg elige la salida Composite
> por defecto en vez de HDMI en la Le Potato.

## Paso 11: Configurar audio

```bash
ssh rostro@<IP_LE_POTATO> 'sudo usermod -aG audio rostro'
```

## Paso 12: Reiniciar y verificar

```bash
ssh rostro@<IP_LE_POTATO> 'sudo reboot'
```

Esperar ~60 segundos y verificar:

```bash
# SSH funciona
ssh rostro@<IP_LE_POTATO> "echo OK"

# Servicios activos
ssh rostro@<IP_LE_POTATO> '
echo "UFW:          $(sudo ufw status | head -1)"
echo "fail2ban:     $(sudo systemctl is-active fail2ban)"
echo "unattended:   $(sudo systemctl is-active unattended-upgrades)"
echo "lightdm:      $(sudo systemctl is-active lightdm)"
echo "app:          $(pgrep -f rostro.main > /dev/null && echo running || echo stopped)"
'
```

Con un monitor HDMI conectado deberia verse el avatar de Rostro.

## Actualizaciones del codigo

Para desplegar cambios nuevos:

```bash
# Desde tu ordenador
rsync -avz \
  --exclude='venv' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.pytest_cache' \
  /ruta/a/rostro/ rostro@<IP_LE_POTATO>:/home/rostro/avatar/

# Reiniciar la app
ssh rostro@<IP_LE_POTATO> 'kill $(pgrep -f rostro.main); sleep 2; cd ~/avatar && DISPLAY=:0 XAUTHORITY=~/.Xauthority nohup bash -c ". venv/bin/activate && python -m rostro.main" > /tmp/rostro.log 2>&1 &'
```

## Troubleshooting

### No se ve nada en la tele (HDMI)
Xorg elige Composite en vez de HDMI por defecto. Verificar:
```bash
ssh rostro@<IP_LE_POTATO> 'DISPLAY=:0 XAUTHORITY=~/.Xauthority xrandr'
```
Activar HDMI manualmente:
```bash
ssh rostro@<IP_LE_POTATO> 'DISPLAY=:0 XAUTHORITY=~/.Xauthority xrandr --output HDMI-1-1 --mode 1920x1080 --primary --output None-1 --off'
```

### La app no arranca al reiniciar
Revisar errores:
```bash
ssh rostro@<IP_LE_POTATO> 'cat ~/.xsession-errors | tail -20'
```
Causa comun: el autostart usa `source` (solo existe en bash, no en sh).
Solucion: usar `. venv/bin/activate` dentro de `bash -c "..."`.

### SSH no conecta despues de reflashear la SD
Limpiar la clave vieja:
```bash
ssh-keygen -R <IP_LE_POTATO>
```

### La placa no responde
Puede haberse reiniciado por un kernel update (unattended-upgrades).
Esperar 1-2 minutos y reintentar. Verificar con ping:
```bash
ping <IP_LE_POTATO>
```

### Error "PortAudio library not found"
Falta libportaudio2:
```bash
ssh rostro@<IP_LE_POTATO> 'sudo apt install -y libportaudio2'
```

## Estructura final en la Le Potato

```
/home/rostro/
├── avatar/
│   ├── venv/               # Entorno virtual Python
│   ├── rostro/              # Codigo fuente
│   ├── config/
│   │   └── default.yaml    # Configuracion
│   ├── assets/
│   │   └── faces/default/  # Face pack
│   ├── .env                 # API keys
│   └── pyproject.toml
└── .config/
    └── openbox/
        └── autostart        # Lanzador automatico
```

## Puertos abiertos

| Puerto | Servicio | Acceso          |
|--------|----------|-----------------|
| 22     | SSH      | Solo red local  |
